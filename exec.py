import zipfile
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""
1. Load the Data: Import the data into Pandas dataframes.
2. Explore and Prepare the Data: Briefly explore the data and prepare it for the model. 
   This usually involves handling missing values, feature engineering, and data normalization.
3. Build the Model: Use a simple classifier, like Logistic Regression, to build the model.
4. Train the Model: Train the model using the training data.
5. Make Predictions: Use the model to make predictions on the test data.
6. Create Submission File: Format the predictions in the same format as 
   gender_submission.csv and output it as a new CSV file.
"""

# Unzipping the file to access its contents
zip_path = '/mnt/data/titanic.zip'
extract_folder = '/mnt/data/titanic'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# Listing the contents of the extracted folder
extracted_files = os.listdir(extract_folder)  # ['gender_submission.csv', 'test.csv', 'train.csv']

# Paths to the files
train_path = os.path.join(extract_folder, 'train.csv')
test_path = os.path.join(extract_folder, 'test.csv')

# Loading the data into Pandas DataFrames
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Displaying the first few rows of each DataFrame
train_head = train_data.head()
test_head = test_data.head()

# Checking for missing values in both training and test datasets
missing_values_train = train_data.isnull().sum()
missing_values_test = test_data.isnull().sum()
"""
Age: 177 missing values.
Cabin: 687 missing values.
Embarked: 2 missing values.
In the test data:

Age: 86 missing values.
Cabin: 327 missing values.
Fare: 1 missing value.
"""

# Handling missing values
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# Dropping columns that require complex feature engineering
train_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Encoding categorical variables
label_encoder = LabelEncoder()
for column in ['Sex', 'Embarked']:
    train_data[column] = label_encoder.fit_transform(train_data[column])
    test_data[column] = label_encoder.transform(test_data[column])

# Checking the transformed data
train_data.head(), test_data.head()

# Splitting the training data into features and target
X = train_data.drop(['Survived'], axis=1)
y = train_data['Survived']

# Splitting data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)

# Training the model
model.fit(X_train, y_train)

# Validating the model
predictions_val = model.predict(X_val)
accuracy = accuracy_score(y_val, predictions_val)

# Making predictions on the test dataset (again for demonstration)
predictions_test = model.predict(test_data)

# Creating the submission DataFrame (again for demonstration)
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions_test
})

# Saving the submission file (again for demonstration)
submission_file_path = '/mnt/data/titanic_submission_recreated.csv'
submission.to_csv(submission_file_path, index=False)
