import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully.")
        print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        return df
    except Exception as e:
        print(" Error loading data:", e)
        return None

df = load_data()


def data_overview(df):
    try:
        print("\n First 5 rows of the dataset:")
        print(df.head())
        print("\n Data Types:")
        print(df.dtypes)
    except Exception as e:
        print(" Error in data overview:", e)

data_overview(df)


def summary_statistics(df):
    try:
        print("\n Summary Statistics:")
        print(df.describe(include='all').T)
    except Exception as e:
        print(" Error in summary statistics:", e)

summary_statistics(df)



def plot_numerical_distribution(df):
    try:
        num_cols = df.select_dtypes(include=['number']).columns
        for col in num_cols:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()
    except Exception as e:
        print(" Error plotting numerical distribution:", e)

plot_numerical_distribution(df)


def plot_categorical_distribution(df):
    try:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            plt.figure(figsize=(6, 4))
            df[col].value_counts().plot(kind='bar')
            plt.title(f"Distribution of {col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(" Error plotting categorical distribution:", e)

plot_categorical_distribution(df)


def correlation_analysis(df):
    try:
        corr = df.select_dtypes(include=['number']).corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title(" Correlation Matrix")
        plt.show()
    except Exception as e:
        print(" Error in correlation analysis:", e)

correlation_analysis(df)

def missing_value_analysis(df):
    try:
        print("\n Missing Value Summary:")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print(" No missing values.")
        else:
            print(missing)
    except Exception as e:
        print(" Error in missing value analysis:", e)

missing_value_analysis(df)

def outlier_detection(df):
    try:
        num_cols = df.select_dtypes(include=['number']).columns
        for col in num_cols:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=df[col])
            plt.title(f" Outlier Detection - {col}")
            plt.show()
    except Exception as e:
        print(" Error in outlier detection:", e)

outlier_detection(df)


