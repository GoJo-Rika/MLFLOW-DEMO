**MLFLOW-DEMO**
================

**Getting Started with MLflow**
------------------------------

### Step 1: Install MLflow

 Install MLflow using pip or conda or uv:
```bash
pip install mlflow
or
conda install -c conda-forge mlflow
or 
uv add mlflow
```
### Step 2: Initialize MLflow

Create a new Jupyter notebook and run the following code to initialize MLflow:
```python
import mlflow

# Initialize MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
```
This sets the tracking URI to `http://127.0.0.1:5000`, which is the default URI for the MLflow UI.

### Step 3: Create an Experiment

Create a new experiment with a name of your choice:
```python
# Create an experiment
experiment_id = mlflow.create_experiment("my_experiment")

print(f"Experiment ID: {experiment_id}")
```
This creates a new experiment and prints the experiment ID.

### Step 4: Log Metrics and Parameters

Log some metrics and parameters to the experiment:
```python
# Log metrics and parameters
with mlflow.start_run(experiment_id=experiment_id):
    mlflow.log_metric("accuracy", 0.9)
    mlflow.log_param("learning_rate", 0.01)
```
This logs the accuracy metric and learning rate parameter to the experiment.

### Setting an Experiment as Default

You can also set an experiment as the default experiment, so that all subsequent `mlflow.start_run()` calls will log to that experiment:
```python
# Set an experiment
mlflow.set_experiment("Check localhost connection")

# Log metrics
with mlflow.start_run():
    mlflow.log_metric("test", 1)
    mlflow.log_metric("random", 2)
```
### Step 5: View the Experiment

View the experiment in the MLflow UI:
```bash
mlflow ui
```
This starts the MLflow UI and displays the experiment.

**Example Use Case**
--------------------

Suppose we want to train a simple linear regression model using scikit-learn and log the model to MLflow. We can do this in a Jupyter notebook:
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv("data/raw/data.csv")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop("target", axis=1), data["target"], test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Log the model to MLflow
with mlflow.start_run(experiment_id=experiment_id):
    mlflow.sklearn.log_model(model, "model")
```
This trains a linear regression model and logs it to MLflow.

**Tips and Variations**

* You can also use `mlflow.set_tracking_uri()` to set the tracking URI to a different location, such as a remote server.
* You can use `mlflow.create_experiment()` to create an experiment with a specific name and description.
* You can use `mlflow.log_artifact()` to log artifacts, such as images or text files, to the experiment.
* You can use `mlflow.search_runs()` to search for runs based on specific criteria, such as metrics or parameters.