import os
import psycopg2
import pandas as pd
from psycopg2 import sql
from psycopg2.extras import execute_values
from data_cleaning import clean_dataset

# Use environment variables for connection
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD")
}

def connect():
    return psycopg2.connect(**DB_CONFIG)

def load_csv_to_db_bulk(csv_path, table_name, conn, date_cols=None):
    print(f"\nLoading '{table_name}' from '{csv_path}'...")

    # Check if table already has data
    cur = conn.cursor()
    cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name)))
    count = cur.fetchone()[0]
    if count > 0:
        print(f"Skipping '{table_name}' — already has {count} rows.")
        return

    # Load and clean data
    df = pd.read_csv(csv_path, parse_dates=date_cols)
    df.columns = [c.lower() for c in df.columns]
    df = clean_dataset(table_name, df)

    # Convert NaT to None for DB compatibility
    df = df.where(pd.notnull(df), None)

    tuples = [tuple(x) for x in df.to_numpy()]
    cols = ', '.join(df.columns)

    query = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
        sql.Identifier(table_name),
        sql.SQL(cols)
    )

    try:
        execute_values(cur, query.as_string(cur), tuples, page_size=1000)
        conn.commit()
        print(f"Inserted {len(df)} rows into '{table_name}'")
    except Exception as e:
        print(f"Error inserting into '{table_name}': {e}")
        conn.rollback()

if __name__ == '__main__':
    conn = connect()

    load_csv_to_db_bulk("data/olist_orders_dataset.csv", "orders", conn, [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ])
    load_csv_to_db_bulk("data/olist_order_items_dataset.csv", "order_items", conn)
    load_csv_to_db_bulk("data/olist_order_payments_dataset.csv", "payments", conn)
    load_csv_to_db_bulk("data/olist_customers_dataset.csv", "customers", conn)
    load_csv_to_db_bulk("data/olist_products_dataset.csv", "products", conn)
    load_csv_to_db_bulk("data/product_category_name_translation.csv", "categories", conn)

    conn.close()
