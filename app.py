import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from io import BytesIO

# Page config
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

# Static logo (small size)
st.sidebar.image("assets/logo.png", width=100)  # <-- your static logo here

# Sidebar title and info
st.sidebar.markdown("---")
st.sidebar.title("Customer Segmentation")
st.sidebar.markdown("Mall Customers Dataset (KMeans Clustering)")


# Load dataset
@st.cache_data
def load_data():
    data_path = "input_data/Mall_Customers.csv"
    df = pd.read_csv(data_path)
    return df

df = load_data()

# Sidebar filters
st.sidebar.subheader("🔍 Filter Customers")
gender = st.sidebar.multiselect("Select Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
age = st.sidebar.slider("Select Age Range", int(df['Age'].min()), int(df['Age'].max()), (20, 50))
income = st.sidebar.slider("Annual Income (k$)", int(df['Annual Income (k$)'].min()), int(df['Annual Income (k$)'].max()), (30, 70))

filtered_df = df[
    (df['Gender'].isin(gender)) &
    (df['Age'].between(age[0], age[1])) &
    (df['Annual Income (k$)'].between(income[0], income[1]))
]

# Title
st.title("🛍️ Mall Customer Segmentation")
st.markdown("This app uses **KMeans clustering** to segment mall customers based on their Age, Income, and Spending Score.")

# Show data
with st.expander("📄 View Raw Data"):
    st.dataframe(filtered_df)

# Elbow method
def plot_elbow(data):
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(range(1, 11), sse, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("SSE (Inertia)")
    ax.set_title("Elbow Method for Optimal k")
    st.pyplot(fig)

# KMeans clustering
def apply_kmeans(data, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans

# PCA plot
def plot_pca(X, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)

    fig, ax = plt.subplots()
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='rainbow', s=50)
    ax.set_title("PCA - Customer Clusters")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    st.pyplot(fig)

# Cluster plots
def plot_clusters(X, labels):
    fig, ax = plt.subplots()
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='rainbow')
    ax.set_title("Customer Clusters")
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])
    st.pyplot(fig)

# Prepare data for clustering
X = filtered_df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
clusters, kmeans_model = apply_kmeans(X, n_clusters=5)
filtered_df['Cluster'] = clusters

# Tabs for output
tab1, tab2, tab3 = st.tabs(["📊 Elbow Curve", "📌 Clustering Output", "🔁 PCA Visualization"])

with tab1:
    st.subheader("📊 Elbow Curve")
    plot_elbow(X)

with tab2:
    st.subheader("📌 Clustered Customers")
    plot_clusters(X[['Annual Income (k$)', 'Spending Score (1-100)']], clusters)
    st.dataframe(filtered_df)

    # Download button
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df(filtered_df)
    st.download_button(
        label="📥 Download Clustered Data as CSV",
        data=csv_data,
        file_name="clustered_customers.csv",
        mime="text/csv",
    )

with tab3:
    st.subheader("🔁 PCA Visualization")
    plot_pca(X, clusters)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Aryan Daiya | Streamlit App for Mall Customer Segmentation")
