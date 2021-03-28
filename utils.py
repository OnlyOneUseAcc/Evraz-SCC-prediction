import pandas as pd

def replace():
    pass


# Время изготовления одного сплава
def date_to_interval(date_obj: pd.Series):
    date = pd.to_datetime(date_obj)
    interval = date.diff()
    interval[1:] = interval[1:].dt.total_seconds() // 60
    interval.name = 'Interval'
    return interval
