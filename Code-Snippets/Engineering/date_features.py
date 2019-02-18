import pandas as pd

#Convert an object to datetime
df['label'] = pd.to_datetime(df.label)

#Create date features
df['year'] = df.pickup_datetime.dt.week #Works with year, month, week, weekday, hour, minute, second...
df.drop(['label'], axis=1, inplace=True)