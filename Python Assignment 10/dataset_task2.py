import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Load the dataset
file_path = "E:/OneDrive/Desktop/SEM1/Programming for Big Data/Assignment/Python Assignment 10/titanic_train.csv"
df = pd.read_csv(file_path)

# Step 2: Handling missing values and outliers
# Fill missing values in 'Age' with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Clip outliers in 'Fare' to a reasonable upper limit (e.g., 95th percentile value)
fare_upper_limit = df['Fare'].quantile(0.95)
df['Fare'] = np.clip(df['Fare'], a_max=fare_upper_limit, a_min=None)

# Step 3: Feature Engineering
# Create 'FamilySize' by combining 'SibSp' and 'Parch'
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Extract titles from 'Name' and create 'Title' feature
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Group rare titles into 'Rare' and map other titles to specific groups
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Step 4:
# Drop columns that are not needed for modeling.
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1, inplace=True)

# Step 5: Data Transformation
# Use ColumnTransformer to apply different preprocessing to different columns.
# For this example, we'll use one-hot encoding for categorical variables 'Sex' and 'Embarked'.
# For numerical features, we'll use StandardScaler.
numerical_features = ['Age', 'Fare', 'FamilySize']
numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])

categorical_features = ['Sex', 'Embarked']
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply the preprocessing to the dataframe
df_transformed = preprocessor.fit_transform(df)
print("Preprocessed Dataset:")
print(df_transformed[:5])
