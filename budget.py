import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.optimize import minimize
import pickle

# Importing data
data = pd.read_csv('advertising_sales_dataset.csv')

# Create DataFrame
df = pd.DataFrame(data)

# Selecting relevant features
columns = ['Instagram', 'Facebook', 'YouTube', 'Twitter', 'TikTok']
X = df[columns]
y = df['Sales']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and fitting the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model and necessary data using pickle
with open('advertising_sales_model.pkl', 'wb') as f:
    pickle.dump((model, X_train, y_train), f)
