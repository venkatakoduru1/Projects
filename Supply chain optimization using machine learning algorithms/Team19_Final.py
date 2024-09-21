import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

import seaborn as sns

input_file_path = r"C:\\Users\\kodur\\OneDrive\\Desktop\\Academic Projects\\FAI_FINAL_PROJ-main\\items.csv"
df_items = pd.read_csv(input_file_path,delimiter='|' )
df_orders = pd.read_csv( r"C:\\Users\\kodur\\OneDrive\\Desktop\\Academic Projects\\FAI_FINAL_PROJ-main\\orders.csv", delimiter='|')

df_items.head()

df_items

df_orders

# Merge DataFrames on 'itemID'
result_df = pd.merge(df_items, df_orders, on='itemID')

# Convert 'time' to datetime format
result_df['time'] = pd.to_datetime(result_df['time'])

# Create a new feature for the day
result_df['day'] = result_df['time'].dt.date

# Group by 'itemID' and 'day' and calculate the cumulative sum of orders for each item on each day
result_df['cumulative_orders_per_item'] = result_df.groupby(['itemID', 'day'])['order'].cumsum()


result_df.cumulative_orders_per_item.value_counts()

# Create bins for customerRating
bins = [0, 3, 4, 5]  # Adjust the bin edges based on your rating distribution
labels = ['low', 'medium', 'high']

result_df['customerRating_category'] = pd.cut(result_df['customerRating'], bins=bins, labels=labels, include_lowest=True)

result_df.drop('customerRating',axis=1)

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming result_df is your DataFrame and it contains an 'orders' column
correlation_matrix = result_df.corr()

# Isolate the correlation values of the 'orders' column
orders_corr = correlation_matrix[['order']]

# Create the heatmap
plt.figure(figsize=(8, 10))
sns.heatmap(orders_corr, annot=True, cmap='coolwarm')
plt.title("Correlation with 'Order' Column")
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.express as px

# Numeric Features
numeric_features = ['recommendedRetailPrice']
result_df[numeric_features].describe()

# Categorical Features
categorical_features = ['customerRating_category']
result_df[categorical_features].value_counts().plot(kind='bar')


# Scatter Plot
sns.scatterplot(x='recommendedRetailPrice', y='cumulative_orders_per_item', data=result_df)
plt.title('Scatter Plot: recommendedRetailPrice vs Cumulative Orders per Item')
plt.show()

# Line Plot
item_450_df = result_df[result_df['itemID'] == 1]
sns.lineplot(x='day', y='cumulative_orders_per_item', data=item_450_df)
plt.title('Line Plot: Cumulative Orders per Item over Time')
plt.xticks(rotation=45)
plt.show()


# Correlation Plot
correlation_matrix = result_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Plot')
plt.show()


# Interactive Line Plot using Plotly
fig = px.line(result_df, x='day', y='cumulative_orders_per_item', color='itemID', title='Interactive Line Plot: Cumulative Orders per Item over Time')
fig.update_xaxes(type='category')  # Set x-axis to categorical
fig.show()

# Distribution Plot for recommendedRetailPrice
sns.histplot(result_df['recommendedRetailPrice'], kde=True)
plt.title('Distribution Plot: recommendedRetailPrice')
plt.show()

# Distribution Plot for salesPrice
sns.histplot(result_df['salesPrice'], kde=True)
plt.title('Distribution Plot: salesPrice')
plt.show()

# Categorical Features
categorical_features = ['customerRating_category']
result_df[categorical_features].value_counts().plot(kind='bar')
plt.title('Bar Plot: Customer Rating Categories')
plt.show()

# Scatter Plot: recommendedRetailPrice vs salesPrice
sns.scatterplot(x='recommendedRetailPrice', y='salesPrice', data=result_df)
plt.title('Scatter Plot: recommendedRetailPrice vs salesPrice')
plt.show()

# Box Plot: recommendedRetailPrice by Customer Rating Category
sns.boxplot(x='customerRating_category', y='recommendedRetailPrice', data=result_df)
plt.title('Box Plot: recommendedRetailPrice by Customer Rating Category')
plt.show()


# Box Plot: salesPrice by Customer Rating Category
sns.boxplot(x='customerRating_category', y='salesPrice', data=result_df)
plt.title('Box Plot: salesPrice by Customer Rating Category')
plt.show()

# Line Plot
sns.lineplot(x='day', y='cumulative_orders_per_item', data=result_df)
plt.title('Line Plot: Cumulative Orders per Item over Time')
plt.xticks(rotation=45)
plt.show()

# Correlation Plot
correlation_matrix = result_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Plot')
plt.show()

# Features (X) and Target variable (y)
features = result_df[['itemID', 'brand', 'manufacturer', 'category1', 'recommendedRetailPrice','salesPrice' ,'customerRating_category']]
target = result_df['cumulative_orders_per_item']

# Convert categorical features to numerical using one-hot encoding
features = pd.get_dummies(features, columns=['customerRating_category'])


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# Standardize the features
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)

# Define a list of regression models
models = [
    DecisionTreeRegressor(),
]

# Train and evaluate each model
for model in models:
    model_name = model.__class__.__name__
    print(f"Training {model_name}...")
    
    # Train the model on standardized features
    model.fit(X_train_standardized, y_train)
    
    # Make predictions on the standardized test set
    predictions = model.predict(X_test_standardized)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
     # Adjusted R-squared
    n = X_test_standardized.shape[0]  # number of samples
    p = X_test_standardized.shape[1]  # number of features
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    print(f'{model_name} Metrics:')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'R-squared (R2): {r2}')
    print(f'Adjusted R-squared: {adjusted_r2}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Root Mean Squared Error (RMSE): {rmse}\n')

    
y_pred = model.predict(X_test)

X_test['predicted_order'] = y_pred
X_test['actual_order'] = y_test

specific_item_data = X_test[X_test['itemID'] == 450]

print(specific_item_data[['predicted_order', 'actual_order']])

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define a deep neural network model
model = Sequential()
model.add(Dense(128, input_dim=X_train_standardized.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# Train the model
model.fit(X_train_standardized, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=2)

# Make predictions on the standardized test set
predictions = model.predict(X_test_standardized).reshape(-1)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)

# Adjusted R-squared
n = X_test_standardized.shape[0]  # number of samples
p = X_test_standardized.shape[1]  # number of features
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print('Deep Neural Network Metrics: ')
print(f'R-squared (R2): {r2}')
print(f'Adjusted R-squared: {adjusted_r2}')

from xgboost import XGBRegressor

# Define an XGBoost regressor model
model_xgb = XGBRegressor()

# Train the model
model_xgb.fit(X_train_standardized, y_train)

# Make predictions on the standardized test set
predictions = model_xgb.predict(X_test_standardized)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)


print('XGBoost Regressor Metrics:')
print(f'R-squared (R2): {r2}')