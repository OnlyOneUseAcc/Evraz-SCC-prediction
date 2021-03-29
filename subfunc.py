import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import combinations


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


def plot_feature_importance(model, data, target_index, title):
    feature_importance = model.estimators_[target_index].feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    fig = plt.figure(figsize=(25, 50))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(data.columns)[sorted_idx])
    plt.title(title)


def normalize_data(data):
    return data.apply(lambda col: (col - col.min()) / (col.max() - col.min()))
