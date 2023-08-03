
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(filepath):
    # Load the dataset into a pandas DataFrame
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return

    # Display the first few rows of the dataset
    print("Preview of the dataset:")
    print(df.head())

    # Get basic information about the dataset
    print("\nDataset Information:")
    print(df.info())

    # Summary statistics of numerical columns
    print("\nSummary Statistics:")
    print(df.describe())

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Visualize the distribution of numerical features using histograms
    print("\nHistograms:")
    df.hist(figsize=(10, 8))
    plt.tight_layout()
    plt.show()

    # Visualize the correlation between numerical features using a heatmap
    print("\nCorrelation Heatmap:")
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.show()

    # Visualize categorical features
    categorical_features = df.select_dtypes(include=["object"]).columns
    for feature in categorical_features:
        print(f"\nValue counts for {feature}:")
        print(df[feature].value_counts())
        sns.countplot(x=feature, data=df)
        plt.xticks(rotation=45)
        plt.show()

if __name__ == "__main__":
    # Replace 'UberDataset.csv' with the actual file name and path
    file_path = r'E:/OneDrive/Desktop/SEM1/Programming for Big Data/Assignment/Python Assignment 10/titanic_train.csv'
    perform_eda(file_path)