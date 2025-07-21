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
- **Data integration** across multiple CSV sources (orders, payments, customers, products)
- **Feature engineering** for RFM segmentation and cohort analysis
- **Interactive dashboard design** using Streamlit and Plotly
- **Machine learning** implementation for churn prediction
- **Visualization of insights** using choropleths and dynamic filters

---

### 🔍 Key Features

#### 📦 Overview Dashboard
- Real-time metrics: revenue, order count, unique customers
- Monthly revenue trends
- Top product categories and payment methods

#### 🧠 Customer Retention Analysis
- Cohort analysis heatmap to track customer retention month-over-month

#### 📌 RFM Segmentation & Clustering
- Generated RFM scores (Recency, Frequency, Monetary)
- Applied **K-Means clustering** to identify customer segments such as:
  - Champions
  - At Risk
  - New Customers
  - Loyal Users
- Interactive scatter plot showing customer distribution

#### 🌍 Geographic Revenue Insights
- Choropleth map visualizing revenue distribution by Brazilian state
- Highlights regional sales patterns and concentration

#### 🔮 Churn Prediction Model
- Built a **Logistic Regression model** to estimate churn probability based on RFM behavior
- Predicted high-risk customers with probabilities and performance metrics

---

### 💡 Business Value

This dashboard serves as a **central decision-support tool** for marketing, sales, and executive teams. It allows the business to:

- Target high-value customer segments
- Identify churn before it happens
- Optimize marketing spend by region
- Align customer lifecycle strategy with actual data

---

### 🧰 Tech Stack

- **Python** (Pandas, Scikit-learn, NumPy)
- **Streamlit** for frontend UI
- **Plotly** for dynamic visualizations
- **GeoJSON** + Choropleth for geographic analysis
- **Machine Learning** for churn prediction and clustering

---

### 📎 Dataset

Used the publicly available [Olist Ecommerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), which includes:

- 100k+ orders from 2016–2018
- Customer, seller, product, and geolocation data
- Payment and review history

---

### 📌 Use Case Fit

This type of dashboard is ideal for:
- Ecommerce startups and DTC brands
- Product managers looking to track retention
- Analysts needing a lightweight, end-to-end BI solution

---
