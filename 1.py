import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import os


#Importing the dataset
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "Titanic-Dataset.csv")
data = pd.read_csv(DATA_PATH)

#Printing the first 5 rows of the dataset
print(data.head())

#printing the last 5 rows of the dataset
print(data.tail())

#Printing the shape of the dataset
print(data.shape)

#printing the information of the dataset
print(data.info())  

#printing the description of the dataset
print(data.describe())

#printting the DAtatypes of the dataset
print(data.dtypes)

#Printing the null values of the dataset
print(data.isnull().sum())

#Filling the NUll values in the dataset with mean only for numerical values
data = data.fillna(data.mean(numeric_only=True))
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Filling the null values in categorical columns with the most frequent value
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])
print(data.info())
print(data.describe())
print(data.isnull().sum())

categorical_values=data.select_dtypes(include=['object']).columns
numerical_values=data.select_dtypes(exclude=['object']).columns
print(categorical_values)
print(numerical_values)


#Encoding the categorical values using LabelEncoder
for i in categorical_values:
    le=LabelEncoder()
    data[i]=le.fit_transform(data[i])

print(data.head())
print(data.describe())
print(data.info())

#Scaling the numerical values using StandardScaler
scaler = MinMaxScaler()
data[numerical_values] = scaler.fit_transform(data[numerical_values])
data[categorical_values] = scaler.fit_transform(data[categorical_values])
print(data[categorical_values].describe())
print(data[numerical_values].describe())
print(data.head())


#Ploting the distribution of the numerical values
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_values):
    plt.subplot(3, 3, i+1)
    sns.boxplot(y=data[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


#Removing the outliers using IQR method
def remove_outliers_iqr(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df
data_cleaned = remove_outliers_iqr(data, numerical_values.tolist() + categorical_values.tolist())

print(f"Original shape: {data.shape}")
print(f"Shape after removing outliers: {data_cleaned.shape}")