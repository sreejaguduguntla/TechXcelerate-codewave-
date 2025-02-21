import flwr as fl
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load data
file_path = "processed_data.csv"
df = pd.read_csv(file_path)

# Separate features and target
X_train = df.drop(columns=["Disease"])  # Features
y_train = df["Disease"]  # Target

# Define model
model = DecisionTreeClassifier()

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config=None):
        return []  # DecisionTreeClassifier does not have traditional parameters like NN

    def set_parameters(self, parameters):
        pass  # No parameters to set for DecisionTreeClassifier

    def fit(self, parameters, config):
        model.fit(X_train, y_train)
        joblib.dump(model, "trained_model.pkl")  # Save trained model
        return self.get_parameters(), len(X_train), {}

    def evaluate(self, parameters, config):
        return 0.0, len(X_train), {}

# Start Flower client
fl.client.start_client(
    server_address="192.168.137.194:8080",
    client=FlowerClient().to_client()
)
