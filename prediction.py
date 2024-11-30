import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
#from main import user_data

# Load the data
data = pd.read_csv("heart_data.csv")

# Preprocess the data (assuming 'target' is the label column)
X = data.drop(columns=['target'])
y = data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

def predict_heart_disease(user_data):
    # Scale the user input data
    user_data = np.array(user_data).reshape(1, -1)
    user_data = scaler.transform(user_data)
    
    # Make a prediction
    prediction = model.predict(user_data)
    
    return prediction[0]

# Example usage (simulating frontend input):
user_data = [51,1,2,94,227,0,0,154,1,0,0,1,3]
prediction_result = predict_heart_disease(user_data)
print(f"Target: {prediction_result}")
# Print age and sex from user_data
age = user_data[0]
sex = user_data[1]
print(f"Age: {age}")
print(f"Sex: {'Male' if sex == 1 else 'Female'}")

if prediction_result == 0:
    print('The Person does not have a Heart Disease')
else:
    print('The Person has Heart Disease')

# Calculate accuracy on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set:", accuracy * 100,"%")
