import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------------------------------------
# Step 1: Calculate RFM Metrics
# ----------------------------------------------------------
def calculate_rfm(df, customer_id='CustomerId', amount_col='Amount', date_col='TransactionStartTime', snapshot_date=None):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if snapshot_date is None:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(customer_id).agg({
        date_col: lambda x: (snapshot_date - x.max()).days,
        amount_col: ['count', 'sum']
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm.reset_index()

    return rfm

# ----------------------------------------------------------
# Step 2: Cluster customers using KMeans
# ----------------------------------------------------------
def cluster_rfm(rfm_df, n_clusters=3, random_state=42):
    features = ['Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm_df[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df['cluster'] = kmeans.fit_predict(scaled)

    return rfm_df

# ----------------------------------------------------------
# Step 3: Assign is_high_risk Label Based on Cluster Summary
# ----------------------------------------------------------
def assign_high_risk_label(rfm_df):
    summary = rfm_df.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].mean()

    # Define risk score (higher recency, lower frequency & monetary → more risky)
    summary['risk_score'] = (
        summary['Recency'].rank(ascending=True) * -1 +
        summary['Frequency'].rank(ascending=True) +
        summary['Monetary'].rank(ascending=True)
    )

    high_risk_cluster = summary['risk_score'].idxmin()

    rfm_df['is_high_risk'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)

    return rfm_df[[ 'CustomerId', 'is_high_risk' ]]

# ----------------------------------------------------------
# Step 4: Merge Label Back into Processed Data
# ----------------------------------------------------------
def label_high_risk_customers(transactions_df, processed_df):
    rfm = calculate_rfm(transactions_df)
    clustered = cluster_rfm(rfm)
    labeled = assign_high_risk_label(clustered)

    # Merge back into processed data
    merged = pd.merge(processed_df, labeled, on='CustomerId', how='left')
    merged['is_high_risk'] = merged['is_high_risk'].fillna(0).astype(int)

    return merged