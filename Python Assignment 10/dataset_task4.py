import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Step 1: Load the dataset
file_path = "E:/OneDrive/Desktop/SEM1/Programming for Big Data/Assignment/Python Assignment 10/titanic_train.csv"
df = pd.read_csv(file_path)

# Step 2: Data Exploration and Visualization

# 1. Bar chart to visualize the distribution of passengers by 'Sex'
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', data=df)
plt.title("Passenger Count by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

# 2. Bar chart to visualize the distribution of passengers by 'Pclass'
plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Count by Pclass")
plt.xlabel("Pclass")
plt.ylabel("Count")
plt.show()

# 3. Scatter plot to visualize the relationship between 'Age' and 'Fare' colored by 'Survived'
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='Fare', data=df, hue='Survived', palette='coolwarm')
plt.title("Age vs. Fare (Colored by Survival)")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.legend(title="Survived", loc='upper right', labels=['Not Survived', 'Survived'])
plt.show()

# Step 3: Remove non-numeric columns before calculating the correlation matrix
numerical_features = df.select_dtypes(include=[int, float])
correlation_matrix = numerical_features.corr()

# 4. Correlation heatmap to visualize the relationships between numerical features
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 5. Box plot to visualize the distribution of fares by passenger class
plt.figure(figsize=(8, 6))
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title("Fare Distribution by Pclass")
plt.xlabel("Pclass")
plt.ylabel("Fare")
plt.show()

# 6. Violin plot to visualize the distribution of ages by sex
plt.figure(figsize=(8, 6))
sns.violinplot(x='Sex', y='Age', data=df)
plt.title("Age Distribution by Sex")
plt.xlabel("Sex")
plt.ylabel("Age")
plt.show()

# 7. Interactive bar chart using Plotly to visualize the distribution of passengers by 'Survived' and 'Pclass'
fig = px.bar(df, x='Pclass', color='Survived', barmode='group', title='Passenger Count by Pclass and Survival',
             labels={'Pclass': 'Passenger Class', 'Survived': 'Survived', 'count': 'Count'})
fig.show()
