import pandas as pd


def replace_comma(data: pd.DataFrame):
    data = data.apply(lambda x: x.apply(str).str.replace(',', '.'))
    data = data.apply(lambda x: pd.to_numeric(x))
    return data


# Время изготовления одного сплава
def date_to_interval(date_obj: pd.Series):
    date = pd.to_datetime(date_obj)
    interval = date.diff()
    interval[1:] = interval[1:].dt.total_seconds() // 60
    interval.name = 'Interval'
    return interval


def drop_columns(data, thresholder):
    data_dropped = data[[column for column in data if data[column].count() / len(data) >= thresholder]]
    return data_dropped
