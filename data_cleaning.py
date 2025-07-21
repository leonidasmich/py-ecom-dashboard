import pandas as pd

def clean_orders(df):
    df = df.dropna(subset=['order_purchase_timestamp'])
    date_cols = [
        'order_approved_at', 
        'order_delivered_carrier_date',
        'order_delivered_customer_date', 
        'order_estimated_delivery_date'
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df = df.dropna(subset=date_cols)
    return df

def clean_order_items(df):
    df = df.dropna(subset=['order_id', 'product_id'])
    df = df.drop_duplicates()
    df['shipping_limit_date'] = pd.to_datetime(df['shipping_limit_date'], errors='coerce')
    df = df.dropna(subset=['shipping_limit_date'])
    return df

def clean_payments(df):
    df = df.dropna(subset=['order_id', 'payment_type'])
    df['payment_type'] = df['payment_type'].str.strip().str.replace(' ', '_').str.lower()
    # Cast numeric fields explicitly
    df['payment_sequential'] = pd.to_numeric(df['payment_sequential'], errors='coerce')
    df['payment_installments'] = pd.to_numeric(df['payment_installments'], errors='coerce')
    df['payment_value'] = pd.to_numeric(df['payment_value'], errors='coerce')
    df = df.dropna()
    return df

def clean_customers(df):
    df = df.dropna(subset=['customer_id'])
    df = df.drop_duplicates()
    return df

def clean_products(df):
    df = df.drop_duplicates(subset=['product_id'])
    df['product_name_lenght'] = pd.to_numeric(df['product_name_lenght'], errors='coerce')
    df['product_description_lenght'] = pd.to_numeric(df['product_description_lenght'], errors='coerce')
    df['product_photos_qty'] = pd.to_numeric(df['product_photos_qty'], errors='coerce')
    df['product_weight_g'] = pd.to_numeric(df['product_weight_g'], errors='coerce')
    df['product_length_cm'] = pd.to_numeric(df['product_length_cm'], errors='coerce')
    df['product_height_cm'] = pd.to_numeric(df['product_height_cm'], errors='coerce')
    df['product_width_cm'] = pd.to_numeric(df['product_width_cm'], errors='coerce')
    df = df.dropna()
    return df

def clean_categories(df):
    df = df.dropna(subset=['product_category_name'])
    df = df.drop_duplicates()
    return df

def clean_dataset(name, df):
    if name == "orders":
        return clean_orders(df)
    elif name == "order_items":
        return clean_order_items(df)
    elif name == "payments":
        return clean_payments(df)
    elif name == "customers":
        return clean_customers(df)
    elif name == "products":
        return clean_products(df)
    elif name == "categories":
        return clean_categories(df)
    else:
        raise ValueError(f"No cleaning function for dataset '{name}'")
