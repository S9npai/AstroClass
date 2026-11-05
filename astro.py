import pathlib
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import joblib as jb
from IPython.core.display_functions import display
from numpy.lib.recfunctions import join_by
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer



class AstroClass():
    def __init__(self):
        datapath = pathlib.Path("magic04.data")
        self.df = pd.read_csv(datapath, header=None)
        self.add_headers()
        self.numeric_features = self.df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_features = self.df.select_dtypes(exclude=np.number).columns.tolist()


    def add_headers(self):
        cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym",
                "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
        self.df.columns = cols


    def inspect(self):
        print("Missing values per column:")
        display(self.df.isnull().sum().sort_values(ascending=False))

        print("\nDuplicated rows:", self.df.duplicated().sum())

        print("\nUnique values per column:")
        display(self.df.nunique().sort_values())

        print("\nDuplicate values per column (non-unique counts):")
        display(self.df.apply(lambda col: col.duplicated().sum()).sort_values(ascending=False))

        print("\nShape:")
        display(self.df.shape)

        print("\nDescription", self.df.info())
        display(self.df.describe(include="all"))


    def feature_dict(self):
        print("Numerical features:", self.numeric_features)
        print("\nCategorical features:", self.categorical_features)


    def display_pairplot(self):
        df_sample = self.df.sample(n=500, random_state=42)
        sns.pairplot(df_sample, hue='class', palette='Set1', diag_kind='kde')
        plt.show()


    def TTsplit(self, test_size=0.2, random_state=42):
        x, y = self.divide()
        return train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=y
        )


    def divide(self):
        df = self.df.copy()
        x = df.drop('class', axis=1)
        y = df["class"].map({'g': 0, 'h': 1})
        return x, y


    def trainer_pipeline(self, x_train, y_train):
        pipeline = Pipeline(steps=[
            ('transformer', PowerTransformer(standardize=True)),
            ('model', LogisticRegression(C=0.8, random_state=42, class_weight={0: 1.2, 1: 0.85}, solver='saga'))
        ])

        pipeline.fit(x_train, y_train)
        return pipeline


    def modelEval(self, pipeline, xtest, ytest):
        y_pred = pipeline.predict(xtest)
        print(classification_report(ytest, y_pred, target_names=['Gamma (0)', 'Hadron (1)']))

        conf_matrix = confusion_matrix(ytest, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Gamma (0)', 'Hadron (1)'],
                    yticklabels=['Gamma (0)', 'Hadron (1)'])
        plt.xlabel('Prediction')
        plt.ylabel('Actual Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()


    def storeWX(self, model, filename):
        joblib.dump(model, filename)




if __name__ == "__main__":
    astro = AstroClass()
    astro.inspect()
    astro.display_pairplot()
    astro.feature_dict()

    x_train, x_test, y_train, y_test = astro.TTsplit()
    model = astro.trainer_pipeline(x_train, y_train)
    astro.modelEval(model, x_test, y_test)

    astro.storeWX(model, "astroWX")
