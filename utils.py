import pandas as pd


def replace_comma(data: pd.DataFrame):
    data = data.apply(lambda x: x.apply(str).str.replace(',', '.'))
    data = data.apply(lambda x: pd.to_numeric(x))
    return data


def date_to_interval(data):
    pass