import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
import pickle
from impyute.imputation.cs import mice
from sklearn.neighbors import LocalOutlierFactor
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel


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


def plot_feature_importance(model, data, title):
    feature_importance = model.feature_importances_
    feat_importance = pd.Series(feature_importance, index=data.columns)
    plt.figure(figsize=(18, 9))
    feat_importance.nlargest(15).plot(kind='barh')
    plt.title(title)
    plt.savefig(f"source/f_i_{title.replace(' ', '_')}.png")


def normalize_data(data):
    return data.apply(lambda col: (col - col.min()) / (col.max() - col.min())
    if (col.max() - col.min()) != 0 else col.max())


def remove_noises(data):
    lof = LocalOutlierFactor(n_neighbors=10, novelty=True)
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

    filter_model = SelectFromModel(regressor, prefit=True, threshold=12e-3)

    feature_indexes = filter_model.get_support()
    feature_names = X.columns[feature_indexes]
    data = pd.DataFrame(filter_model.transform(X), columns=feature_names, index=X.index)
    print(f'удалили признаков: {X.shape[1] - data.shape[1]}')
    data[Y.columns] = Y
    return data, pd.Series(regressor.feature_importances_, index=X.columns)
