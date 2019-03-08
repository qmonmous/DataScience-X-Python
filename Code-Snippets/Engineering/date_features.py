import pandas as pd

#Convert an object to datetime
df['column'] = pd.to_datetime(df.column)

#Create date features
df['year'] = df.pickup_datetime.dt.week #Works with year, month, week, weekday, hour, minute, second...
df.drop(['column'], axis=1, inplace=True)