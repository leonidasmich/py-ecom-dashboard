import streamlit as st
import pandas as pd
import plotly.express as px
import json
import urllib.request
import os
import psycopg2
from psycopg2 import sql
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data_prep import get_connection


st.set_page_config(page_title="Ecommerce Dashboard", layout="wide")
st.title("🛒 Ecommerce Analytics Dashboard")

# PostgreSQL connection
@st.cache_resource
def get_conn():
    return get_connection()

# Load data from PostgreSQL
@st.cache_data
def get_data():
    query = """
    SELECT o.order_id, o.order_purchase_timestamp, o.customer_id, c.customer_unique_id, c.customer_state,
           p.payment_value, oi.product_id, pr.product_category_name,
           cat.product_category_name_english, p.payment_type
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN payments p ON o.order_id = p.order_id
    JOIN order_items oi ON o.order_id = oi.order_id
    LEFT JOIN products pr ON oi.product_id = pr.product_id
    LEFT JOIN categories cat ON pr.product_category_name = cat.product_category_name
    """
    conn = get_conn()
    df = pd.read_sql(query, conn)
    conn.close()
    return df

df = get_data()
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

# Sidebar filters
st.sidebar.header("Filters")
min_date = df['order_purchase_timestamp'].min().date()
max_date = df['order_purchase_timestamp'].max().date()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

states = st.sidebar.multiselect("Customer States", sorted(df['customer_state'].unique()), sorted(df['customer_state'].unique()))
categories = st.sidebar.multiselect("Product Categories", sorted(df['product_category_name_english'].dropna().unique()), sorted(df['product_category_name_english'].dropna().unique()))

# Apply filters
df = df[(df['order_purchase_timestamp'].dt.date >= date_range[0]) &
        (df['order_purchase_timestamp'].dt.date <= date_range[1])]
df = df[df['customer_state'].isin(states)]
df = df[df['product_category_name_english'].isin(categories)]

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🧠 Customer Insights", "📌 RFM Analysis", "🌐 Geo Insights", "🔮 Prediction"])

# Overview Tab
with tab1:
    st.subheader("KPIs")
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Revenue", f"R${df['payment_value'].sum():,.2f}")
    col2.metric("📦 Total Orders", df['order_id'].nunique())
    col3.metric("👥 Unique Customers", df['customer_unique_id'].nunique())

    st.markdown("---")
    st.subheader("📈 Revenue Over Time")
    df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M').dt.to_timestamp()
    monthly_revenue = df.groupby('order_month')['payment_value'].sum().reset_index()
    st.plotly_chart(px.line(monthly_revenue, x="order_month", y="payment_value"), use_container_width=True)

    st.subheader("🏷️ Top 10 Product Categories")
    top_cats = df['product_category_name_english'].value_counts().nlargest(10).reset_index()
    top_cats.columns = ["Category", "Orders"]
    st.plotly_chart(px.bar(top_cats, x="Orders", y="Category", orientation="h"), use_container_width=True)

    st.subheader("💳 Payment Type Distribution")
    pay_types = df['payment_type'].value_counts().reset_index()
    pay_types.columns = ["Payment Type", "Count"]
    st.plotly_chart(px.pie(pay_types, names="Payment Type", values="Count"), use_container_width=True)

# Customer Insights Tab
with tab2:
    st.subheader("📊 Cohort Analysis")
    df['cohort_month'] = df.groupby('customer_unique_id')['order_purchase_timestamp'].transform('min').dt.to_period('M').dt.to_timestamp()
    df['cohort_index'] = ((df['order_month'].dt.year - df['cohort_month'].dt.year) * 12 +
                          (df['order_month'].dt.month - df['cohort_month'].dt.month) + 1)
    cohort_data = df.groupby(['cohort_month', 'cohort_index']) \
                    .agg(n_customers=('customer_unique_id', 'nunique')) \
                    .reset_index()
    cohort_pivot = cohort_data.pivot(index='cohort_month', columns='cohort_index', values='n_customers')
    cohort_normalized = cohort_pivot.divide(cohort_pivot.iloc[:, 0], axis=0)
    cohort_normalized.columns = cohort_normalized.columns.astype(str)
    cohort_normalized = cohort_normalized.fillna(0).round(2)
    cohort_normalized.index = cohort_normalized.index.strftime('%Y-%m')
    st.dataframe(cohort_normalized, use_container_width=True)

# RFM Analysis Tab
with tab3:
    st.subheader("📌 RFM Segmentation & Clustering")

    snapshot_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'nunique',
        'payment_value': 'sum'
    }).rename(columns={
        'order_purchase_timestamp': 'Recency',
        'order_id': 'Frequency',
        'payment_value': 'Monetary'
    }).reset_index()

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    cluster_labels = {
        0: "Loyal",
        1: "Champions",
        2: "At Risk",
        3: "New Customers"
    }
    rfm['Segment'] = rfm['Cluster'].map(cluster_labels)

    st.markdown("### 🧠 Segment Overview")
    seg_counts = rfm['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Customer Count']
    st.dataframe(seg_counts, use_container_width=True)

    st.markdown("### 📈 Clustered Scatter Plot")
    fig = px.scatter(rfm, x='Recency', y='Frequency', size='Monetary',
                     color='Segment', hover_data=['customer_unique_id'],
                     title="RFM Segmentation with K-Means")
    st.plotly_chart(fig, use_container_width=True)

# Geo Insights Tab
with tab4:
    st.subheader("🌎 Revenue by State")

    state_rev = df.groupby('customer_state')['payment_value'].sum().reset_index()
    state_rev.columns = ['uf', 'revenue']

    geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    with urllib.request.urlopen(geojson_url) as response:
        brazil_geo = json.load(response)

    fig = px.choropleth(
        state_rev,
        geojson=brazil_geo,
        locations='uf',
        featureidkey='properties.sigla',
        color='revenue',
        color_continuous_scale='Viridis',
        scope='south america',
        labels={'revenue': 'Revenue'},
        title="💰 Revenue by Brazilian State"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        height=700,
        width=1000,
        margin={"r": 0, "t": 50, "l": 0, "b": 0}
    )
    st.plotly_chart(fig, use_container_width=True)

# Prediction Tab
with tab5:
    st.subheader("🔮 Churn Prediction (RFM-based)")

    snapshot_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'nunique',
        'payment_value': 'sum'
    }).rename(columns={
        'order_purchase_timestamp': 'Recency',
        'order_id': 'Frequency',
        'payment_value': 'Monetary'
    }).reset_index()

    rfm['Churn'] = (rfm['Recency'] > 180).astype(int)

    X = rfm[['Recency', 'Frequency', 'Monetary']]
    y = rfm['Churn']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rfm_test = rfm.iloc[y_test.index].copy()
    rfm_test['Churn_Prob'] = model.predict_proba(X_test)[:, 1]

    st.markdown("### 📉 Sample Churn Probabilities")
    st.dataframe(
        rfm_test[['customer_unique_id', 'Recency', 'Frequency', 'Monetary', 'Churn_Prob']]
        .sort_values('Churn_Prob', ascending=False)
        .head(10),
        use_container_width=True
    )

    st.markdown("### 📋 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(2)
    st.dataframe(report_df, use_container_width=True)