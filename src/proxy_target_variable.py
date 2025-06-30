import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_rfm_features(df):
    try:
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)

        rfm = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Value': 'sum'
        }).reset_index()

        rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
        return rfm
    except Exception as e:
        print("Error creating RFM features:", e)
        return None

def scale_rfm_features(rfm):
    try:
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
        return rfm_scaled
    except Exception as e:
        print("Error scaling RFM features:", e)
        return None

def cluster_customers(rfm_scaled, rfm, n_clusters=3):
    """Cluster customers using KMeans."""
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(rfm_scaled)
        rfm['Cluster'] = clusters
        return rfm
    except Exception as e:
        print(f"Error clustering customers: {e}")
        return None
def analyze_clusters(rfm):
    try:
        # Calculate mean Recency, Frequency, Monetary per cluster
        cluster_profile = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        
        # Add count of customers per cluster
        cluster_profile['Count'] = rfm['Cluster'].value_counts().sort_index()
        
        # Identify high risk cluster: highest Recency, lowest Frequency and Monetary
        high_risk_cluster = cluster_profile.sort_values(
            ['Recency', 'Frequency', 'Monetary'], ascending=[False, True, True]
        ).index[0]
        
        # print(cluster_profile)
        # print(f"High risk cluster identified: {high_risk_cluster}")
        
        return cluster_profile, high_risk_cluster
    
    except Exception as e:
        print("Error analyzing clusters:", e)
        return None, None


def identify_high_risk_cluster(rfm):
    """Identify the high-risk cluster based on RFM means."""
    try:
        cluster_profile = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        # Sort by Recency DESC, Frequency ASC, Monetary ASC to find high-risk cluster
        high_risk_cluster = cluster_profile.sort_values(
            ['Recency', 'Frequency', 'Monetary'], ascending=[False, True, True]).index[0]
        return high_risk_cluster
    except Exception as e:
        print(f"Error identifying high-risk cluster: {e}")
        return None

def assign_high_risk_label(rfm, high_risk_cluster):
    """Create the is_high_risk column."""
    try:
        rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
        return rfm[['CustomerId', 'is_high_risk']]
    except Exception as e:
        print(f"Error assigning high-risk label: {e}")
        return None

def merge_with_main_data(df, rfm_labels):
    """Merge the high-risk labels back into the main dataframe."""
    try:
        df_merged = df.merge(rfm_labels, on='CustomerId', how='left')
        print("Data merged with high-risk labels successfully.")
        return df_merged
    except Exception as e:
        print(f"Error merging data: {e}")
        return None