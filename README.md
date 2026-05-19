# 📊 Ecommerce Analytics Dashboard – Case Study

### 🧠 Business Problem

An ecommerce business struggles to understand key customer behaviors and sales trends across regions. While data exists in silos (orders, customers, products), the company lacks a **unified analytics tool** to track performance, identify loyal customers, and anticipate churn.

---

### 🎯 Objective

To design and build a **data-driven analytics dashboard** that allows business stakeholders to:

- Monitor revenue and order performance in real time  
- Understand customer lifecycle and retention  
- Segment customers based on behavior (RFM)  
- Visualize geographic sales distribution  
- Predict which customers are likely to churn  

---

### 🛠 My Role

I led the full project lifecycle:
- **Data ingestion** from structured CSV datasets into a normalized **PostgreSQL** database
- Developed an **automated data seeding pipeline** with bulk inserts and conflict resolution
- **Containerized the database** using Docker for consistent local and remote environments
- Designed and deployed an interactive **Streamlit dashboard** with responsive filtering
- Implemented **cohort analysis**, **RFM segmentation**, and **KMeans clustering**
- Developed a **churn prediction model** using Logistic Regression

---

### 🔍 Key Features

#### 📦 Overview Dashboard
- Real-time KPIs: Total Revenue, Orders, Unique Customers  
- Monthly revenue time series chart  
- Top product categories by order volume  
- Payment method distribution pie chart  

#### 🧠 Customer Retention Analysis
- Cohort matrix showing month-over-month retention  
- Normalized heatmap to highlight drop-offs and loyalty trends  

#### 📌 RFM Segmentation & Clustering
- Recency, Frequency, and Monetary scoring for each customer  
- Applied **K-Means Clustering** to define segments:
  - Champions  
  - Loyal  
  - New Customers  
  - At Risk  
- Visualized segments in a bubble scatterplot  

#### 🌍 Geographic Revenue Insights
- Choropleth map using GeoJSON to show revenue by Brazilian state  
- Highlights regional disparities and potential market focus areas  

#### 🔮 Churn Prediction Model
- Simulated churn based on recency threshold  
- Trained **Logistic Regression** on RFM data  
- Predicted churn probabilities with ranking  
- Displayed classification report with precision and recall  

---

### 🔐 Deployment & Data Architecture Highlights

- **PostgreSQL** hosted via Docker for structured data storage  
- Bulk-insert logic with row count validation to avoid duplicates  
- Environment variables securely managed with **Streamlit Secrets Manager**  
- Streamlit connects directly to the PostgreSQL instance using `psycopg2`  
- Dashboard auto-refreshes with caching via `@st.cache_data` and `@st.cache_resource`

---

### 📎 Dataset

Used the publicly available [Olist Brazilian Ecommerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), which includes:

- 100k+ orders from 2016–2018  
- Product, customer, seller, payment, and review details  
- Rich geolocation and category metadata  

---

### 💡 Business Value

This dashboard provides a **single source of truth** for ecommerce analytics, enabling the business to:

- Track high-level performance across time and geography  
- Prioritize marketing efforts toward loyal and high-spending customers  
- Identify at-risk customers before they churn  
- Align internal teams with actionable, real-time data  

---

### 🧰 Tech Stack

- **Python**: Pandas, Scikit-learn, NumPy, psycopg2  
- **PostgreSQL**: Relational database for normalized data  
- **Streamlit**: Interactive analytics frontend  
- **Plotly**: Rich data visualizations and maps  
- **React + Highcharts**: Browser-based dashboard preview with reusable chart cards
- **Docker**: Containerized backend infrastructure  
- **GeoJSON**: Spatial mapping of revenue insights  
- **Streamlit Secrets**: Secure credential management for cloud deployment  

---

### ⚛️ React Dashboard Preview

The repository also includes a Vite-powered React UI that renders the ecommerce analytics with Highcharts.

```bash
npm install
npm run generate:data
npm run dev
```

For a production check:

```bash
npm run build
```

---

### 📌 Use Case Fit

This solution is ideal for:
- Ecommerce brands and marketplaces  
- BI analysts and product teams  
- Data professionals building lightweight, full-stack analytics apps  
- Startups seeking end-to-end visibility without enterprise BI tools  

---
