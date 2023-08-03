import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Load the dataset
file_path = "E:/OneDrive/Desktop/SEM1/Programming for Big Data/Assignment/Python Assignment 10/titanic_train.csv"
df = pd.read_csv(file_path)

# Step 2: Preprocessing (You can reuse the preprocessing steps from the previous code)

# Check if the 'Title' column exists before dropping it
if 'Title' in df.columns:
    df.drop('Title', axis=1, inplace=True)

# Step 3: Split the data into training and testing sets
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature Scaling (not required for all models, but we'll use it for SVM)
# Drop non-numeric columns before scaling
X_train_numeric = X_train.select_dtypes(include=[int, float])
X_test_numeric = X_test.select_dtypes(include=[int, float])

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_numeric)
X_test_imputed = imputer.transform(X_test_numeric)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Step 5: Implement the models
# Model 1: Support Vector Machine (SVM)
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Model 2: Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_imputed, y_train) # Use X_train_imputed instead of X_train_numeric

# Step 6: Evaluate the models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

svm_accuracy, svm_precision, svm_recall, svm_f1 = evaluate_model(svm_model, X_test_scaled, y_test)
rf_accuracy, rf_precision, rf_recall, rf_f1 = evaluate_model(rf_model, X_test_imputed, y_test) # Use X_test_imputed instead of X_test_numeric

# Step 7: Compare the performance
print("Support Vector Machine (SVM) Performance:")
print(f"Accuracy: {svm_accuracy:.2f}")
print(f"Precision: {svm_precision:.2f}")
print(f"Recall: {svm_recall:.2f}")
print(f"F1 Score: {svm_f1:.2f}")

print("\nRandom Forest Performance:")
print(f"Accuracy: {rf_accuracy:.2f}")
print(f"Precision: {rf_precision:.2f}")
print(f"Recall: {rf_recall:.2f}")
print(f"F1 Score: {rf_f1:.2f}")
