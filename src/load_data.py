import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataeda(file_path):
    try:
        df = pd.read_csv(file_path)
        print(" Data loaded successfully.")
        return df
    except Exception as e:
        print(" Error loading data:", e)
        return None

def data_overview(df):
    print("\n Dataset Overview")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nData Types:\n", df.dtypes)

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

def clean_raw_data(df):
    # 1. Drop duplicate rows
    df = df.drop_duplicates()

    # 2. Remove columns with all missing values
    df = df.dropna(axis=1, how='all')

    # 3. Handle missing values (example: fill with mean for numeric, mode for categorical)
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # 4. Standardize column names (optional: lower case, no spaces)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # 5. Convert date columns (if known or guessed)
    for col in df.columns:
        if 'date' in col or 'time' in col:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass

    # 6. Remove non-numeric characters from numeric columns (if any)
    for col in df.select_dtypes(include='object').columns:
        if df[col].str.replace('.', '', 1).str.isnumeric().all():
            df[col] = pd.to_numeric(df[col])

    # 7. Optional: Remove outliers or invalid entries
    # Example: remove rows where Amount < 0
    if 'amount' in df.columns:
        df = df[df['amount'] >= 0]

    return df