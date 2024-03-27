import pandas as pd 
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

class Processor:
    def __init__(self, model):
        self.model=model

    def train_test_splitter(self, df, target, test_size=0.2, random_state=42):
        y = df[target]
        X = df.drop([target], axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def train(self, df, target='target', train_test_spit=True):
        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        y_train = df[target]
        X_train = df.drop([target], axis = 1)
        self.model.fit(X_train, y_train, cat_cols)

    def evaluate(self, df, target='target'):
        if target in df.columns():
            df = df.drop([target], axis=1)
        preds = self.model.predict(df)
        df[target] = preds
        return df
