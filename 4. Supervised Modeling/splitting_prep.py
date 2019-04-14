X = df['add_all_features']
y = df['add_target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Make sure the split went well
#X_train.shape, y_train.shape, X_test.shape, y_test.shape