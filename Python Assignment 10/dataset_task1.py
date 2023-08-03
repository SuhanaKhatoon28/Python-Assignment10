import pandas as pd
import numpy as np

# Step 1: Load the dataset
file_path = "E:/OneDrive/Desktop/SEM1/Programming for Big Data/Assignment/Python Assignment 10/titanic_train.csv"
df = pd.read_csv(file_path)

# Step 2: Handling missing values
# For this example, we'll fill the missing values in 'Age' with the mean and 'Embarked' with the mode.
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Step 3: Encoding categorical variables
# Convert 'Sex' and 'Embarked' columns into numerical values using one-hot encoding.
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Step 4: Feature Engineering
# We can create new features from existing ones. For example, we can combine 'SibSp' and 'Parch' to create 'FamilySize'.
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Step 5: Drop unnecessary columns
# If certain columns are not needed, we can drop them.
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1, inplace=True)

# Display the preprocessed dataset
print("Preprocessed Dataset:")
print(df.head())
