import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
import pickle
from impyute.imputation.cs import mice
from sklearn.neighbors import LocalOutlierFactor
import os
import re
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error
import catboost
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def get_scores(models, y_test, X_test):
    r2_scores = np.zeros(3)
    mse_scores = np.zeros(3)
    mae_scores = np.zeros(3)
    mape_scores = np.zeros(3)
    params = []
    params.append(None)
    for ix, model in enumerate(models):
        r2_scores[ix] = r2_score(y_test, models[model].predict(X_test))
        mse_scores[ix] = mean_squared_error(y_test, models[model].predict(X_test))
        mae_scores[ix] = mean_absolute_error(y_test, models[model].predict(X_test))
        mape_scores[ix] = MAPE(y_test.to_numpy(), models[model].predict(X_test))
        if ix > 0:
            params.append(models[model].get_params())
    scores = pd.DataFrame(index=models.keys(), columns=['R2', 'MAE', 'MSE', 'MAPe'])
    scores['R2'] = r2_scores
    scores['MAE'] = mae_scores
    scores['MSE'] = mse_scores
    scores['MAPe'] = mape_scores
    scores['Best params'] = params
    return scores

# Время изготовления одного сплава
def date_to_interval(date_obj: pd.Series):
    date = pd.to_datetime(date_obj).copy()
    interval = date.diff()
    interval[1:] = interval[1:].dt.total_seconds() // 60
    interval[0] = float(interval.mode())
    interval.name = 'Interval'
    interval = pd.to_numeric(interval)
    return interval


def drop_columns(data: pd.DataFrame, threshold=0.85):
    data_dropped = data[[column for column in data if data[column].count() / len(data) >= threshold]]
    dropped_col = list()
    for col in data.columns:
        if col not in data_dropped.columns:
            dropped_col.append(col)

    print(f"Удаленные колонки: {', '.join(dropped_col)}")
    return data_dropped, dropped_col


def plot_feature_importance(model, data, title, dir):
    feature_importance = model.feature_importances_
    feat_importance = pd.Series(feature_importance, index=data.columns)
    plt.figure(figsize=(18, 9))
    feat_importance.nlargest(15).plot(kind='barh')
    plt.title(title)
    plt.savefig(f"../../source/{dir}f_i_{title.replace(' ', '_')}.png")


def normalize_data(data):
    return data.apply(lambda col: (col - col.min()) / (col.max() - col.min())
    if (col.max() - col.min()) != 0 else col.max())


def remove_noises(data):
    lof = LocalOutlierFactor(n_neighbors=15, novelty=True)
    lof.fit(data)
    outlier_predicted = lof.predict(data)
    clear_data = data[outlier_predicted == 1].copy()

    return clear_data


def show_result(predicted, target, path=None):
    fig, axes = plt.subplots(4, figsize=(15, 30))
    keys = list(target.keys())
    for index in range(len(keys)):
        axes[index].set_title(keys[index])
        axes[index].plot(np.linspace(0, predicted.iloc[::5, index].shape[0],
                                     num=predicted.iloc[::5, index].shape[0]),
                         target[keys[index]][::5],
                         label='Целевая'
                         )
        axes[index].plot(np.linspace(0, predicted.iloc[::5, index].shape[0],
                                     num=predicted.iloc[::5, index].shape[0]),
                         predicted.iloc[::5, predicted.columns.to_list().index(keys[index])],
                         label='Предсказанная', alpha=0.75)
        axes[index].legend(borderpad=1, shadow=True, bbox_to_anchor=(1, 1))
    if path is not None:
        plt.savefig(path)


def models_save(models):
    for key in models.keys():
        with open(f'model/{key}.pkl', 'wb') as f:
            pickle.dump(models[key], f)


def models_load():
    models = {}
    for root, dirs, files in os.walk("../model/", topdown=False):
        for file_name in files:
            with open(os.path.join('../model/', file_name), 'rb') as file:
                models[file_name.split('.')[0]] = pickle.load(file)
    return models


def fill_empty_values(data: pd.DataFrame):
    imputed_training = mice(data.values)
    empty_mask = data.isna()
    data_array = data.values
    data_array[empty_mask] = imputed_training[empty_mask]
    return pd.DataFrame(data_array,
                        columns=data.columns,
                        index=data.index)


def select_target(X_train, X_test, y_train, y_test):
    X_train_dataset = {}
    X_test_dataset = {}
    y_train_dataset = {}
    y_test_dataset = {}

    X_test[y_test.columns] = y_test
    X_train[y_train.columns] = y_train

    combs = combinations(y_train.columns, 3)
    for comb in combs:
        target = list(set(y_train.columns) - set(comb))[0]
        y_train_dataset[target] = y_train[target]
        y_test_dataset[target] = y_test[target]
        X_test_dataset[target] = X_test.drop(columns=target)
        X_train_dataset[target] = X_train.drop(columns=target)

    return X_train_dataset, X_test_dataset, y_train_dataset, y_test_dataset


def filter_features(X: pd.DataFrame, Y: pd.DataFrame):
    regressor = ExtraTreesRegressor(n_estimators=350, n_jobs=2, random_state=21)
    regressor.fit(X, Y)

    filter_model = SelectFromModel(regressor, prefit=True, threshold=3e-3)

    feature_indexes = filter_model.get_support()
    feature_names = X.columns[feature_indexes]
    data = pd.DataFrame(filter_model.transform(X), columns=feature_names, index=X.index)
    print(f'удалили признаков: {X.shape[1] - data.shape[1]}')
    data[Y.columns] = Y
    return data, pd.Series(regressor.feature_importances_, index=X.columns)


def fill_future_columns(X_train, X_test, y_train, y_test, future_column):
    future_column_list = future_column.copy()
    while len(future_column_list) != 0:
        prediction_result = {}
        for column in future_column_list:
            cur_eval_data = catboost.Pool(X_test, y_test[column])
            cur_model = catboost.CatBoostRegressor(iterations=1000,
                                                   learning_rate=0.05,
                                                   task_type='GPU',
                                                   loss_function='MAE',
                                                   random_seed=42)
            cur_model.fit(X_train, y_train[column],
                          use_best_model=True,
                          early_stopping_rounds=100,
                          eval_set=cur_eval_data,
                          silent=True)
            prediction_result[column] = [cur_model.copy(), r2_score(
                y_pred=cur_model.predict(X_test), y_true=y_test[column])]
        best_r2 = -20
        best_target = ''
        for key in prediction_result.keys():
            if prediction_result[key][1] > best_r2:
                best_r2 = prediction_result[key][1]
                best_target = key
        if prediction_result[best_target][1] > 0.5:  # перестаем добавлять новые признаки если плохо предсказываем их
            print(f'for column {best_target}: {best_r2}')
            X_train[best_target] = prediction_result[best_target][0].predict(X_train)
            X_test[best_target] = prediction_result[best_target][0].predict(X_test)
        else:
            break
        future_column_list.remove(best_target)
    return X_train, X_test


def MAPE(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape


def get_report(models_dict, target_columns_list, X_test, y_test):
    r2_scores = np.zeros(4)
    mae_scores = np.zeros(4)
    mape_scores = np.zeros(4)
    mse_scores = np.zeros(4)
    for itarget, target in enumerate(target_columns_list):
        r2_scores[itarget] = r2_score(y_true=y_test[target],
                                      y_pred=models_dict[target].predict(X_test))
        mse_scores[itarget] = mean_squared_error(y_true=y_test[target],
                                                 y_pred=models_dict[target].predict(X_test))
        mape_scores[itarget] = MAPE(y_test[target],
                                    models_dict[target].predict(X_test))
        mae_scores[itarget] = mean_absolute_error(y_true=y_test[target],
                                                  y_pred=models_dict[target].predict(X_test))

    scores = pd.DataFrame(index=target_columns_list, columns=['R2', 'MAE', 'MSE', 'MAPe'])
    scores['R2'] = r2_scores
    scores['MAE'] = mae_scores
    scores['MSE'] = mse_scores
    scores['MAPe'] = mape_scores
    return scores


def save_obj(obj, path):
    with open(f'{path}.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(f'{path}.pkl', 'rb') as f:
        return pickle.load(f)


def to_categorical(df, target, q, path):
    cat_target = pd.qcut(df[target[0]], q)
    mask = {cat: i for i, cat in enumerate(cat_target.unique())}
    cat_target = cat_target.replace(mask)
    save_obj(mask, path)
    return cat_target


def to_numerical(targets, dict):
    numerical_targets = []
    for i, target in enumerate(targets):
        for key in dict.keys():
            if target == dict[key]:
                numerical_targets.append(key.right - abs(key.left))
    return np.array(numerical_targets)


def split_substnce(data, elements_data):
    aug_data = data.copy()
    for col in elements_data.columns:
        elements = elements_data[col][elements_data[col] != 0].index
        part_of_name_col = re.match(r'([а-я]|[А-Я]|\s)+', col)
        if part_of_name_col is None:
            continue
        for element in elements:
            name_col = f'{part_of_name_col.group(0)} {element}'
            if name_col in aug_data.columns:
                aug_data[name_col] += aug_data[col] * elements_data.loc[element, col]
            else:
                aug_data[name_col] = aug_data[col] * elements_data.loc[element, col]
    return aug_data
