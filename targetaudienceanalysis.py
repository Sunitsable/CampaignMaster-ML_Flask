# import pandas as pd
# data = pd.read_csv("SocialMediaUsersDataset.csv")
# data.head()
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import mean_squared_error
# import numpy as np

# # Define the file path
# file_path = 'SocialMediaUsersDataset.csv'

# # Define chunk size
# chunk_size = 1000

# # Initialize a dictionary to store counts of predictions
# all_predictions = []
# all_actuals = []

# # Iterate over the dataset in chunks
# for chunk in pd.read_csv(file_path, chunksize=chunk_size):
#     # Convert categorical variables to numerical values
#     label_encoders = {}
#     for column in ['Gender', 'City', 'Country']:
#         if column not in label_encoders:
#             label_encoders[column] = LabelEncoder()
#         chunk[column] = label_encoders[column].fit_transform(chunk[column])

#     # Extract features from 'DOB'
#     chunk['Age'] = pd.Timestamp('now').year - pd.to_datetime(chunk['DOB']).dt.year

#     # Select features and target variable
#     X = chunk[['Gender', 'Age', 'City', 'Country']]
#     y = chunk['Interests']

#     # Train a Random Forest Classifier
#     clf = RandomForestClassifier()
#     clf.fit(X, y)

#     # Make predictions for the current chunk
#     y_pred = clf.predict(X)

#     # Append the predictions and actuals to the list
#     all_predictions.extend(y_pred)
#     all_actuals.extend(y)

# # Convert the lists to numpy arrays
# all_predictions = np.array(all_predictions)
# all_actuals = np.array(all_actuals)

# # Calculate the MSE
# totalWrongPrediction = 0
# for i in range(len(all_predictions)):
#   if all_actuals[i]!=all_predictions[i]:
#     totalWrongPrediction = totalWrongPrediction + 1

# print(f'Mean Squared Error: {totalWrongPrediction}/{len(all_predictions)}')

# # Define and fit label encoders
# label_encoders = {}
# for column in ['Gender', 'City', 'Country']:
#     label_encoders[column] = LabelEncoder()
#     label_encoders[column].fit(data[column])  # Assuming 'data' is your original dataset
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder

# import numpy as np


# user_info = {
#     'Gender': 'Male',
#     'DOB': '2001-11-11',
#     'City': 'Colombia',
#     'Country': 'West Indies'
# }


# # Convert categorical variables to numerical values
# user_info_encoded = {}
# for column in ['Gender', 'City', 'Country']:
#     user_info_encoded[column] = label_encoders[column].transform(np.array([user_info[column]]))[0] if user_info[column] in label_encoders[column].classes_ else -1


# # Extract features from 'DOB'
# user_info_encoded['Age'] = pd.Timestamp('now').year - pd.to_datetime(user_info['DOB']).year

# # Select features for the user
# X_user = np.array([[user_info_encoded['Gender'], user_info_encoded['Age'], user_info_encoded['City'], user_info_encoded['Country']]])

# # Make prediction for the user
# user_prediction = clf.predict(X_user)[0]

# print(f'Predicted Interests for the User: {user_prediction}')
#import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import pandas as pd
data = pd.read_csv("SocialMediaUsersDataset.csv")
data.head()

# Define the file path
file_path = 'SocialMediaUsersDataset.csv'

# Define chunk size
chunk_size = 1000

# Initialize a dictionary to store counts of predictions
all_predictions = []
all_actuals = []

# Iterate over the dataset in chunks
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    # Convert categorical variables to numerical values
    label_encoders = {}
    for column in ['Gender', 'City', 'Country']:
        if column not in label_encoders:
            label_encoders[column] = LabelEncoder()
        chunk[column] = label_encoders[column].fit_transform(chunk[column])

    # Extract features from 'DOB'
    chunk['Age'] = pd.Timestamp('now').year - pd.to_datetime(chunk['DOB']).dt.year

    # Select features and target variable
    X = chunk[['Gender', 'Age', 'City', 'Country']]
    y = chunk['Interests']

    # Train a Random Forest Classifier
    clf = RandomForestClassifier()
    clf.fit(X, y)

    # Make predictions for the current chunk
    y_pred = clf.predict(X)

    # Append the predictions and actuals to the list
    all_predictions.extend(y_pred)
    all_actuals.extend(y)

# Convert the lists to numpy arrays
all_predictions = np.array(all_predictions)
all_actuals = np.array(all_actuals)

# Calculate the MSE
totalWrongPrediction = np.sum(all_actuals != all_predictions)
totalPredictions = len(all_predictions)
mean_squared_error = totalWrongPrediction / totalPredictions

print(f'Mean Squared Error: {totalWrongPrediction}/{totalPredictions}')

# Define and fit label encoders
label_encoders = {}
for column in ['Gender', 'City', 'Country']:
    label_encoders[column] = LabelEncoder()
    label_encoders[column].fit(data[column])  # Assuming 'data' is your original dataset

# Save the trained model and label encoders using pickle
with open('social_media_model.pkl', 'wb') as f:
    pickle.dump((clf, label_encoders), f)

# Example usage for predicting interests for a user
user_info = {
    'Gender': 'Male',
    'DOB': '2001-11-11',
    'City': 'Colombia',
    'Country': 'West Indies'
}

# Load the trained model and label encoders
with open('social_media_model.pkl', 'rb') as f:
    clf, label_encoders = pickle.load(f)

# Convert categorical variables to numerical values for user info
user_info_encoded = {}
for column in ['Gender', 'City', 'Country']:
    user_info_encoded[column] = label_encoders[column].transform(np.array([user_info[column]]))[0] if user_info[column] in label_encoders[column].classes_ else -1

# Extract features from 'DOB'
user_info_encoded['Age'] = pd.Timestamp('now').year - pd.to_datetime(user_info['DOB']).year

# Select features for the user
X_user = np.array([[user_info_encoded['Gender'], user_info_encoded['Age'], user_info_encoded['City'], user_info_encoded['Country']]])

# Make prediction for the user
user_prediction = clf.predict(X_user)[0]

print(f'Predicted Interests for the User: {user_prediction}')
