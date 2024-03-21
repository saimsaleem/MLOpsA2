from flask import Flask, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

app = Flask(__name__)

# Load the Iris dataset
data_path = "iris.data"
column_names = ["feature1", "feature2", "feature3", "feature4", "label"]
df = pd.read_csv(data_path, names=column_names)

# Preprocess the data
X = df.drop('label', axis=1)
y = df['label']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Define a Flask endpoint to get the classification report
@app.route('/', methods=['GET'])
def get_classification_report():
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # Generate the classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    return jsonify(report)

if __name__ == '__main__':
    app.run(debug=True)


    
