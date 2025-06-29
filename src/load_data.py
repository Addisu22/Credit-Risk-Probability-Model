import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_overview(df):
    print(" Dataset Overview")
    # print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\n Data Types:\n", df.dtypes)

def summary_statistics(df):
    print("\n Summary Statistics")
    display(df.describe(include='all'))

def plot_numerical_distributions(df, num_cols):
    print("\n Distribution of Numerical Features")
    df[num_cols].hist(bins=30, figsize=(15, 10))
    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(df, cat_cols):
    print("\n Distribution of Categorical Features")
    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.xticks(rotation=45)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()

def correlation_analysis(df, num_cols):
    print("\n Correlation Analysis")
    corr = df[num_cols].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def check_missing_values(df):
    print("\n Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
    else:
        print(missing)

def detect_outliers(df, num_cols):
    print("\n Outlier Detection via Box Plots")
    for col in num_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.show()

