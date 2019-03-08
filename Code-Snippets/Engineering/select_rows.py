#Select df's 5th row, or row = 4 with row starting at 0
df[4:5]

#With an iterator
df.iloc[i]


df = df.loc[(df['column'] == "value")


X = df.loc[:, ['feature1', 'feature2']]
y = df.loc[:, 'target']