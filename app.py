import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import urllib.request
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from data_prep import get_connection


st.set_page_config(page_title="Ecommerce Dashboard", layout="wide")
st.title("🛒 Ecommerce Analytics Dashboard")


HIGHCHARTS_CDN = "https://code.highcharts.com"


def render_highcharts(options, height=420, modules=None):
    render_highcharts_component(
        options=options,
        height=height,
        core_script=f"{HIGHCHARTS_CDN}/highcharts.js",
        constructor="chart",
        modules=modules,
    )


def render_highcharts_map(options, map_data, height=650):
    render_highcharts_component(
        options=options,
        height=height,
        core_script=f"{HIGHCHARTS_CDN}/maps/highmaps.js",
        constructor="mapChart",
        modules=[],
        map_data=map_data,
    )


def render_highcharts_component(
    options,
    height,
    core_script,
    constructor,
    modules=None,
    map_data=None,
):
    options_json = json.dumps(options, allow_nan=False)
    map_data_json = json.dumps(map_data, allow_nan=False) if map_data else "null"
    module_scripts = "\n".join(
        f'<script src="{HIGHCHARTS_CDN}/{module}"></script>'
        for module in (modules or [])
    )

    components.html(
        f"""
        <div id="container" style="height: {height}px; width: 100%;"></div>
        <script src="{core_script}"></script>
        {module_scripts}
        <script src="{HIGHCHARTS_CDN}/modules/accessibility.js"></script>
        <script>
          Highcharts.setOptions({{
            lang: {{
              thousandsSep: ','
            }}
          }});

          const chartOptions = {options_json};
          const mapData = {map_data_json};
          if (mapData && chartOptions.series && chartOptions.series.length > 0) {{
            chartOptions.series[0].mapData = mapData;
          }}

          Highcharts.{constructor}('container', chartOptions);
        </script>
        """,
        height=height + 30,
    )


@st.cache_data
def get_brazil_geojson():
    geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    with urllib.request.urlopen(geojson_url) as response:
        brazil_geo = json.load(response)

    for feature in brazil_geo.get("features", []):
        properties = feature.setdefault("properties", {})
        properties["uf"] = properties.get("sigla")

    return brazil_geo


# PostgreSQL connection
@st.cache_resource
def get_conn():
    return get_connection()

# Load data from PostgreSQL
@st.cache_data
def get_data():
    # cache busted
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

if len(date_range) != 2:
    st.info("Select a start and end date to render the dashboard.")
    st.stop()

states = st.sidebar.multiselect("Customer States", sorted(df['customer_state'].unique()), sorted(df['customer_state'].unique()))
categories = st.sidebar.multiselect("Product Categories", sorted(df['product_category_name_english'].dropna().unique()), sorted(df['product_category_name_english'].dropna().unique()))

# Apply filters
df = df[(df['order_purchase_timestamp'].dt.date >= date_range[0]) &
        (df['order_purchase_timestamp'].dt.date <= date_range[1])]
df = df[df['customer_state'].isin(states)]
df = df[df['product_category_name_english'].isin(categories)]
df = df.copy()

if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M').dt.to_timestamp()

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
    monthly_revenue = df.groupby('order_month')['payment_value'].sum().reset_index()
    render_highcharts(
        {
            "chart": {"type": "line", "backgroundColor": "transparent"},
            "title": {"text": None},
            "xAxis": {
                "categories": monthly_revenue['order_month'].dt.strftime('%Y-%m').tolist(),
                "title": {"text": "Order Month"},
            },
            "yAxis": {
                "title": {"text": "Revenue (R$)"},
                "labels": {"format": "R${value:,.0f}"},
            },
            "tooltip": {
                "shared": True,
                "valuePrefix": "R$",
                "valueDecimals": 2,
            },
            "credits": {"enabled": False},
            "series": [
                {
                    "name": "Revenue",
                    "data": monthly_revenue['payment_value'].round(2).astype(float).tolist(),
                    "color": "#2f7ed8",
                }
            ],
        }
    )

    st.subheader("🏷️ Top 10 Product Categories")
    top_cats = df['product_category_name_english'].value_counts().nlargest(10).reset_index()
    top_cats.columns = ["Category", "Orders"]
    top_cats = top_cats.sort_values("Orders")
    render_highcharts(
        {
            "chart": {"type": "bar", "backgroundColor": "transparent"},
            "title": {"text": None},
            "xAxis": {
                "categories": top_cats["Category"].tolist(),
                "title": {"text": None},
            },
            "yAxis": {
                "min": 0,
                "title": {"text": "Orders"},
                "allowDecimals": False,
            },
            "tooltip": {"pointFormat": "<b>{point.y:,.0f}</b> orders"},
            "credits": {"enabled": False},
            "series": [
                {
                    "name": "Orders",
                    "data": top_cats["Orders"].astype(int).tolist(),
                    "color": "#00a6a6",
                }
            ],
        }
    )

    st.subheader("💳 Payment Type Distribution")
    pay_types = df['payment_type'].value_counts().reset_index()
    pay_types.columns = ["Payment Type", "Count"]
    render_highcharts(
        {
            "chart": {"type": "pie", "backgroundColor": "transparent"},
            "title": {"text": None},
            "tooltip": {"pointFormat": "<b>{point.percentage:.1f}%</b> ({point.y:,.0f})"},
            "plotOptions": {
                "pie": {
                    "allowPointSelect": True,
                    "cursor": "pointer",
                    "dataLabels": {
                        "enabled": True,
                        "format": "{point.name}: {point.percentage:.1f}%",
                    },
                }
            },
            "credits": {"enabled": False},
            "series": [
                {
                    "name": "Payments",
                    "colorByPoint": True,
                    "data": [
                        {"name": row["Payment Type"], "y": int(row["Count"])}
                        for _, row in pay_types.iterrows()
                    ],
                }
            ],
        }
    )

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
    cohort_heatmap = [
        {
            "x": x_index,
            "y": y_index,
            "value": float(value * 100),
            "cohort": cohort_month,
            "period": period,
        }
        for y_index, (cohort_month, row) in enumerate(cohort_normalized.iterrows())
        for x_index, (period, value) in enumerate(row.items())
    ]
    render_highcharts(
        {
            "chart": {"type": "heatmap", "backgroundColor": "transparent"},
            "title": {"text": None},
            "xAxis": {
                "categories": cohort_normalized.columns.tolist(),
                "title": {"text": "Months Since First Purchase"},
            },
            "yAxis": {
                "categories": cohort_normalized.index.tolist(),
                "title": {"text": "Cohort Month"},
                "reversed": True,
            },
            "colorAxis": {
                "min": 0,
                "max": 100,
                "stops": [
                    [0, "#f7fbff"],
                    [0.5, "#6baed6"],
                    [1, "#08306b"],
                ],
            },
            "legend": {"align": "right", "layout": "vertical", "verticalAlign": "middle"},
            "tooltip": {
                "pointFormat": "Cohort <b>{point.cohort}</b><br/>Month <b>{point.period}</b><br/>Retention <b>{point.value:.0f}%</b>"
            },
            "plotOptions": {
                "series": {
                    "dataLabels": {
                        "enabled": True,
                        "format": "{point.value:.0f}%",
                        "style": {"textOutline": "none"},
                    }
                }
            },
            "credits": {"enabled": False},
            "series": [{"name": "Retention", "borderWidth": 1, "data": cohort_heatmap}],
        },
        height=520,
        modules=["modules/heatmap.js"],
    )
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
    bubble_series = []
    for segment, segment_df in rfm.groupby('Segment'):
        bubble_series.append(
            {
                "name": segment,
                "data": [
                    {
                        "x": int(row["Recency"]),
                        "y": int(row["Frequency"]),
                        "z": round(float(row["Monetary"]), 2),
                        "name": row["customer_unique_id"],
                    }
                    for _, row in segment_df.iterrows()
                ],
            }
        )

    render_highcharts(
        {
            "chart": {"type": "bubble", "plotBorderWidth": 1, "zoomType": "xy", "backgroundColor": "transparent"},
            "title": {"text": "RFM Segmentation with K-Means"},
            "xAxis": {
                "title": {"text": "Recency (days)"},
                "gridLineWidth": 1,
            },
            "yAxis": {
                "title": {"text": "Frequency"},
                "allowDecimals": False,
            },
            "tooltip": {
                "pointFormat": "Customer: <b>{point.name}</b><br/>Recency: <b>{point.x}</b> days<br/>Frequency: <b>{point.y}</b><br/>Monetary: <b>R${point.z:,.2f}</b>"
            },
            "plotOptions": {
                "bubble": {
                    "minSize": "3%",
                    "maxSize": "18%",
                    "opacity": 0.72,
                }
            },
            "credits": {"enabled": False},
            "series": bubble_series,
        },
        height=520,
        modules=["highcharts-more.js"],
    )

# Geo Insights Tab
with tab4:
    st.subheader("🌎 Revenue by State")

    state_rev = df.groupby('customer_state')['payment_value'].sum().reset_index()
    state_rev.columns = ['uf', 'revenue']
    brazil_geo = get_brazil_geojson()
    render_highcharts_map(
        {
            "chart": {"backgroundColor": "transparent"},
            "title": {"text": "💰 Revenue by Brazilian State"},
            "mapNavigation": {
                "enabled": True,
                "buttonOptions": {"verticalAlign": "bottom"},
            },
            "colorAxis": {
                "min": 0,
                "stops": [
                    [0, "#e5f5e0"],
                    [0.5, "#74c476"],
                    [1, "#00441b"],
                ],
            },
            "tooltip": {"pointFormat": "{point.name}: <b>R${point.value:,.2f}</b>"},
            "credits": {"enabled": False},
            "series": [
                {
                    "data": [
                        [row["uf"], round(float(row["revenue"]), 2)]
                        for _, row in state_rev.iterrows()
                    ],
                    "keys": ["uf", "value"],
                    "joinBy": "uf",
                    "name": "Revenue",
                    "states": {"hover": {"color": "#a4edba"}},
                    "dataLabels": {"enabled": True, "format": "{point.properties.sigla}"},
                }
            ],
        },
        brazil_geo,
    )

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