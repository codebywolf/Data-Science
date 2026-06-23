import pandas as pd
from sqlalchemy import create_engine

# Create MySQL connection
engine = create_engine("mysql+pymysql://root:root@localhost/olist_db")

datasets_names = [
    'olist_customers_dataset.csv',
    'olist_order_payments_dataset.csv',
    'olist_products_dataset.csv',
    'olist_geolocation_dataset.csv',
    'olist_order_reviews_dataset.csv',
    'olist_sellers_dataset.csv',
    'olist_order_items_dataset.csv',
    'olist_orders_dataset.csv',
    'product_category_name_translation.csv'
]

tables = {}

for name in datasets_names:
    # Handle the translation file
    if name.startswith('product_'):
        clean_key = 'product_category_name_translation'
    else:
        # Strip 'olist_' from the start and '_dataset.csv' from the end
        clean_key = name.replace('olist_', '').replace('_dataset.csv', '')

    # Store key: path pair safely without collisions
    tables[clean_key] = f"../datasets/{name}"

# Execute migration pipeline straight into MySQL
for table_name, file_path in tables.items():
    df = pd.read_csv(file_path)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"Table '{table_name}' loaded successfully")