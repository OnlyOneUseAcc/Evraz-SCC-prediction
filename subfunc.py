import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations
import pickle

from impyute.imputation.cs import mice


def replace_comma(data: pd.DataFrame):
    data = data.apply(lambda x: x.apply(str).str.replace(',', '.'))
    data = data.apply(lambda x: pd.to_numeric(x))
    return data


# Время изготовления одного сплава
def date_to_interval(date_obj: pd.Series):
    date = pd.to_datetime(date_obj).copy()
    interval = date.diff()
    interval[1:] = interval[1:].dt.total_seconds() // 60
    interval[0] = float(interval.mode())
    interval.name = 'Interval'
    return interval


def drop_columns(data: pd.DataFrame, threshold=0.85):
    data_dropped = data[[column for column in data if data[column].count() / len(data) >= threshold]]
    dropped_col = list()
    for col in data.columns:
        if col not in data_dropped.columns:
            dropped_col.append(col)

    print(f"Удаленные колонки: {', '.join(dropped_col)}")
    return data_dropped


def drop_rows(data: pd.DataFrame, threshold=0.35):
    nan_count_per_row = data.isnull().sum(axis=1)
    indexes_to_delete = []
    for index in nan_count_per_row.index:
        if nan_count_per_row.iloc[index] / data.shape[1] > threshold:
            indexes_to_delete.append(index)

    return data.drop(indexes_to_delete, axis='index')


def mean_filling(data: pd.DataFrame, usefull_list, group_list):
    full_data = data.copy()

    for count in range(1, len(group_list)):
        for comb in combinations(group_list, count):
            full_data[usefull_list] = full_data.groupby(list(comb))[usefull_list].transform(
                lambda column: column.fillna(column.mean()))

    full_data = full_data.fillna(full_data.mean(axis=0))

    return full_data


def plot_feature_importance(model, data, title):
    feature_importance = model.feature_importances_
    feat_importance = pd.Series(feature_importance, index=data.columns)
    plt.figure(figsize=(10, 5))
    feat_importance.nlargest(15).plot(kind='barh')
    plt.title(title)
    plt.savefig(f'source/{title}.png')


def normalize_data(data):
    return data.apply(lambda col: (col - col.min()) / (col.max() - col.min())
    if (col.max() - col.min()) != 0 else col.max())


def remove_noises(data):
    clear_data = data.copy()
    for col in clear_data:
        std = clear_data[col].std()
        mean = clear_data[col].mean()
        noise_indexes = clear_data[col][abs(clear_data[col]) > (mean + 3 * std)].index
        clear_data = clear_data.drop(noise_indexes, axis='index')

    return clear_data


def show_result(predicted, target, path=None):
    t_columns = target.columns
    target = target.to_numpy()
    for i in range(predicted.shape[1]):
        plt.figure(figsize=(50, 10))
        plt.title(t_columns[i])
        plt.plot(np.linspace(0, predicted[::10].shape[0], num=predicted[::10].shape[0]),
                 target[::10, i], c='b', label='Целевая'
                 )

        plt.plot(np.linspace(0, predicted[::10].shape[0], num=predicted[::10].shape[0]),
                 predicted[::10, i], c='g', label='Предсказанная')
        plt.legend()

        if path is not None:
            plt.savefig(f'{path}{i}.png')


def multi_model_save(models):
    for target in models.keys():
        with open(f'model/{target}.pkl', 'wb') as f:
            pickle.dump(models[target], f)


def multi_model_load():
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def fill_empty_values(data: pd.DataFrame):
    imputed_training = mice(data.values)
    empty_mask = data.isna()
    data_array = data.values
    data_array[empty_mask] = imputed_training[empty_mask]
    print(data_array)
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
        X_test_dataset[target] = X_test.drop(columns = target)
        X_train_dataset[target] = X_train.drop(columns = target)

    return X_train_dataset, X_test_dataset, y_train_dataset, y_test_dataset
