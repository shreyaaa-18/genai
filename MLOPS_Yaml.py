#config.yaml:
project_name: YAML_MLOPS_Practical

model:
  algorithm: RandomForest
  n_estimators: 100
  max_depth: 5
  random_state: 42

training:
  test_size: 0.2

# train.py:
import yaml
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load YAML configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

print("Project:", config["project_name"])

# Step 2: Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target



# Step 3: Split data using YAML settings
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["training"]["test_size"],
    random_state=config["model"]["random_state"]
)

# Step 4: Create model using YAML parameters
model = RandomForestClassifier(
    n_estimators=config["model"]["n_estimators"],
    max_depth=config["model"]["max_depth"],
    random_state=config["model"]["random_state"]
)

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Algorithm:", config["model"]["algorithm"])
print("Number of trees:", config["model"]["n_estimators"])
print("Test data proportion:", config["training"]["test_size"])
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Step 7: Save the trained model
joblib.dump(model, "model.pkl")
print("Model saved successfully as model.pkl")
