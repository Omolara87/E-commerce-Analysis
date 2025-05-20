import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

# Load purchase history
purchases = pd.read_csv("purchase_history.csv")
purchases['purchase_date'] = pd.to_datetime(purchases['purchase_date'])

# Simulate SQL DB for products and campaigns
products_df = pd.DataFrame({
    'product_id': [201, 202, 203, 204, 205, 206, 207],
    'product_name': ['Laptop', 'Phone', 'Mouse', 'Monitor', 'Keyboard', 'Tablet', 'Headphones'],
    'category': ['Electronics'] * 7,
    'price': [1000, 700, 30, 200, 50, 300, 100]
})
campaign_df = pd.DataFrame({
    'campaign_id': [301, 302, 303],
    'campaign_name': ['Holiday Sale', 'Back to School', 'Clearance'],
    'start_date': pd.to_datetime(['2023-12-01', '2024-01-15', '2024-02-10']),
    'end_date': pd.to_datetime(['2023-12-31', '2024-01-31', '2024-03-01'])
})

# Load simulated SQL
conn = sqlite3.connect(':memory:')
products_df.to_sql('products', conn, index=False, if_exists='replace')
campaign_df.to_sql('campaigns', conn, index=False, if_exists='replace')
products_sql = pd.read_sql("SELECT * FROM products", conn)

# Merge data
data = purchases.merge(products_sql, on='product_id', how='left')

# Handle missing values and outliers
data['purchase_amount'].fillna(data['purchase_amount'].median(), inplace=True)
data['purchase_amount'] = np.clip(data['purchase_amount'], data['purchase_amount'].quantile(0.01), data['purchase_amount'].quantile(0.99))

# Feature Engineering
today = pd.to_datetime('2024-04-01')
recency = data.groupby('customer_id')['purchase_date'].max().reset_index()
recency['recency_days'] = (today - recency['purchase_date']).dt.days

frequency = data.groupby('customer_id')['purchase_id'].count().reset_index().rename(columns={'purchase_id': 'frequency'})
monetary = data.groupby('customer_id')['purchase_amount'].sum().reset_index().rename(columns={'purchase_amount': 'monetary'})

# Merge RFM features
rfm = recency.merge(frequency, on='customer_id').merge(monetary, on='customer_id')

# Calculate CLV (basic version: average value * frequency)
rfm['clv'] = (rfm['monetary'] / rfm['frequency']) * rfm['frequency']

# Customer segmentation
rfm['segment'] = pd.qcut(rfm['clv'], 4, labels=['Low', 'Medium', 'High', 'Top'])

# Pie chart of customer segments
plt.figure(figsize=(6, 6))
rfm['segment'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Customer Segments by CLV')
plt.ylabel('')
plt.show()

# Bar chart of top-selling products
top_products = data['product_name'].value_counts().head(5)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_products.index, y=top_products.values)
plt.title('Top-Selling Products')
plt.xlabel('Product')
plt.ylabel('Units Sold')
plt.show()

# Line plot of sales over time
sales_trend = data.groupby('purchase_date')['purchase_amount'].sum()
plt.figure(figsize=(10, 5))
sales_trend.plot()
plt.title('Sales Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()

# Export results
rfm.to_csv("customer_segments.csv", index=False)
data.to_csv("processed_purchases.csv", index=False)
