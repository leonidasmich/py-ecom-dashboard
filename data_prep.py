import pandas as pd


def load_data():
    orders = pd.read_csv("data/olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
    order_items = pd.read_csv("data/olist_order_items_dataset.csv")
    payments = pd.read_csv("data/olist_order_payments_dataset.csv")
    customers = pd.read_csv("data/olist_customers_dataset.csv")
    products = pd.read_csv("data/olist_products_dataset.csv")
    categories = pd.read_csv("data/product_category_name_translation.csv")

    merged = orders.merge(order_items, on="order_id") \
                   .merge(payments, on="order_id") \
                   .merge(customers, on="customer_id") \
                   .merge(products, on="product_id", how="left") \
                   .merge(categories, on="product_category_name", how="left")

    return merged


