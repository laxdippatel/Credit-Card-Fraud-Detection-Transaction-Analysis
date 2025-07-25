# Fraud_detection_analysis.py

import pandas as pd
import numpy as np

# 🎯 Step 1: Simulate Sample Data

np.random.seed(42)

data = {
    'TransactionID': range(1001, 1011),
    'CustomerID': ['C001', 'C002', 'C003', 'C001', 'C004', 'C005', 'C002', 'C003', 'C006', 'C002'],
    'Merchant': ['Amazon', 'Flipkart', 'Myntra', 'Amazon', 'Zomato', 'Swiggy', 'Flipkart', 'Myntra', 'Amazon', 'Flipkart'],
    'Amount': [1500, 25000, 1200, 1490, 560, 670, 30000, 1100, 900, 40000],
    'Timestamp': pd.date_range(start='2025-07-01 10:00', periods=10, freq='H'),
    'Status': ['Success', 'Success', 'Failed', 'Success', 'Success', 'Failed', 'Success', 'Success', 'Success', 'Success']
}

df = pd.DataFrame(data)
df['IsFraud'] = [0, 1, 0, 0, 0, 0, 1, 0, 0, 1]

# 🔍 Step 2: Clean the Data (Remove 'Failed')
df_clean = df[df['Status'] == 'Success'].reset_index(drop=True)



# 📊 Step 3: Exploratory Data Analysis & Features

# 1. Total Spent by Each Customer
total_customer = df_clean.groupby('CustomerID')['Amount'].sum().reset_index()

# 2. Average Spend per Merchant
avg_by_merchant = df_clean.groupby('Merchant')['Amount'].mean().reset_index().sort_values(by='Amount', ascending=False)

# 3. Highest Single Transaction
highest_txn = df_clean[df_clean['Amount'] == df_clean['Amount'].max()]

# 4. Transaction Count by Merchant
merchant_count = df_clean['Merchant'].value_counts().reset_index()
merchant_count.columns = ['Merchant', 'TransactionCount']

# 5. Day-wise Total Spending
df_clean['date'] = df_clean['Timestamp'].dt.date
daywise_spent = df_clean.groupby('date')['Amount'].sum().reset_index()

# 6. Hourly Spending Pattern
df_clean['Hour'] = df_clean['Timestamp'].dt.hour
hourly_spend = df_clean.groupby('Hour')['Amount'].sum().reset_index()

# 7. Customer-wise Average Transaction Amount
customer_avg = df_clean.groupby('CustomerID')['Amount'].mean().reset_index()
customer_avg.columns = ['CustomerID', 'AverageAmount']

# 8. Transactions Above ₹10,000
hightxn = df_clean[df_clean['Amount'] > 10000]

# 9. Fraud Rate by Merchant
fraud_rate = df_clean.groupby('Merchant')['IsFraud'].mean().reset_index().sort_values(by='FraudRate', ascending=False)

# 10. Total Amount Lost to Fraud
fraud_amount = df_clean[df_clean['IsFraud'] == 1]['Amount'].sum()

# 11. GST Column (18% charged)
df_clean['GST_18%'] = df_clean['Amount'] * 0.18

# 12. Normalized Amount Column (zero mean)
df_clean['NormalizedAmount'] = df_clean['Amount'] - df_clean['Amount'].mean()

# 13. Amount in USD (₹1 = $0.012)
df_clean['Amount_USD'] = df_clean['Amount'] * 0.012

# 14. Fraud by Customer
fraud_by_customer = df_clean[df_clean['IsFraud'] == 1].groupby('CustomerID').size().reset_index(name='FraudCount')
