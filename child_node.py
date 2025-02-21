import flwr as fl
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import pickle

# Load preprocessed data
df = pd.read_csv("processed_data.csv")
X_train = df.drop(columns=["Disease"])
y_train = df["Disease"]

# Define model
model = DecisionTreeClassifier()

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config=None):
        """Send model parameters to the server."""
        return []  # DecisionTreeClassifier doesn't have traditional parameters

    def set_parameters(self, parameters):
        """Receive global model parameters from the server (not needed for DecisionTree)."""
        pass

    def fit(self, parameters, config):
        """Train model locally on hospital data."""
        model.fit(X_train, y_train)
        joblib.dump(model, "trained_model.pkl")  # Save locally trained model
        return self.get_parameters(), len(X_train), {}

    def evaluate(self, parameters, config):
        """Dummy evaluation method required by Flower."""
        return 0.0, len(X_train), {}

# Start the client with the correct method
if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="192.168.137.194:8080",  # ✅ Replace with actual main node IP
        client=FlowerClient()
    )

    server_address="192.168.137.194:8080",
    client=FlowerClient().to_client()
)
