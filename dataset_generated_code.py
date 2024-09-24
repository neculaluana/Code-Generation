
import pandas as pd
import re

example_df = pd.DataFrame({'duration': ['7 year', '2day', '4 week', '8 month']},
                          index=list(range(1,5)))

def f(df=example_df):
    df[['number', 'time']] = df.duration.str.extract(r'(\d+)(\D+)', expand=True)
    df['time_days'] = df['time'].replace({'year': 365,'month': 30, 'day': 1, 'week': 7}, regex=True)
    return df
