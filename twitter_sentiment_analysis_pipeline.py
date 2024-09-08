# ---------- Imports and Libraries ----------
import pandas as pd
from datetime import datetime
import seaborn as sns
from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
# --------------------------------------------------


# ---------- Display settings.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
filterwarnings('ignore')

# ---------- Contains feature engineering steps.
def feature_engineering(df):

    def _GetTimePeriod(date):
        hour = date.hour
        if 22 <= hour or hour < 2:
            return '22:00-02:00'
        elif 2 <= hour < 6:
            return '02:00-06:00'
        elif 6 <= hour < 10:
            return '06:00-10:00'
        elif 10 <= hour < 14:
            return '10:00-14:00'
        elif 14 <= hour < 18:
            return '14:00-18:00'
        else:
            return '18:00-22:00'

    def _GetSeason(date):
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'


    df = df.dropna()

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('GMT')
    df['date'] = df['date'].dt.tz_convert('Etc/GMT-3')

    df['season'] = df['date'].apply(_GetSeason)

    df["day"] = [date.strftime('%A') for date in df["date"]]

    df['4_intervals'] = df['date'].apply(_GetTimePeriod)

    df['label'] = df['label'].map({1: 'positive', -1: 'negative', 0: 'neutral'})

    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    numeric_features = df.select_dtypes(include=['number']).columns.tolist()

    print("-----Target Label Analysis-----")
    print(df['label'].value_counts())
    print("---------------------------")
    print(100 * df["label"].value_counts() / len(df))
    print("---------------------------")

    plt.figure(figsize=(8, 6))
    df['label'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Label Frequency Distribution')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.show()

    colors = ['green', 'orange', 'blue']
    labels = ['positive', 'negative', 'neutral']
    values = df['label'].value_counts() / df['label'].shape[0]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=40,
                      marker=dict(colors=colors, line=dict(color='black', width=5)))
    fig.update_layout(
        title_text="label")
    fig.show()

    return df


# ---------- It performs data cleaning, building the Logistic Regression model, and sentiment prediction in tweets.
def data_preparation_and_logistic_regression(df):

    df['tweet'] = df['tweet'].str.lower()
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['tweet'])

    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.2, random_state=5)
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2}")

    df2 = pd.read_csv("tweets_21.csv")
    df2['date'] = pd.to_datetime(df2['date'], errors='coerce')
    df2['tweet'] = df2['tweet'].str.lower()

    X_tfidf_21 = tfidf_vectorizer.transform(df2['tweet'])
    df2['label'] = model.predict(X_tfidf_21)

    print("*****35 elements in the tail*****")
    print(df2.tail(35))

    return df, df2



def main():
    labeled_dataframe = pd.read_csv("tweets_labeled.csv")
    labeled_dataframe = feature_engineering(labeled_dataframe)
    labeled_dataframe, tweets_dataframe = data_preparation_and_logistic_regression(labeled_dataframe)



if __name__ == "__main__":
    print("The process has started.")
    main()