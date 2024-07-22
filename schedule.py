

import pandas as pd
import pickle

# Read the data
data = pd.read_csv('sentimentdataset.csv')

# Convert data to DataFrame
df = pd.DataFrame(data)

# Preprocess the 'Timestamp' column to extract hour
df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour

# Preprocess the 'Platform' column to remove leading and trailing whitespace
df['Platform'] = df['Platform'].str.strip()

# Group data by platform and hour, count the number of unique users for each group
platform_hourly_active_users = df.groupby(['Platform', 'Hour'])['User'].nunique()

# Pickle the processed DataFrame and grouped data
with open('sentiment_data.pkl', 'wb') as f:
    pickle.dump((df, platform_hourly_active_users), f)

# Print the top 10 best times for each platform directly without using a function
# Load the processed DataFrame and grouped data
with open('sentiment_data.pkl', 'rb') as f:
    df, platform_hourly_active_users = pickle.load(f)

# Find the top 10 best times for each platform
platforms = df['Platform'].unique()
for platform in platforms:
    platform_data = platform_hourly_active_users[platform].sort_values(ascending=False).head(10)
    print(f"Top 10 Best Times for {platform}:")
    for hour in platform_data.index.tolist():
        print(f"Hour: {hour}")
