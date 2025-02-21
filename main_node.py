import flwr as fl
import pickle
from sklearn.tree import DecisionTreeClassifier

class CustomStrategy(fl.server.strategy.FedAvg):
    """Custom strategy to save the global model after training rounds."""

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, _ = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            print(f"✅ Round {rnd} - Model Aggregated Successfully!")

            # Save the model after the final training round
            if rnd == 3:  # Ensure this matches num_rounds
                print("💾 Saving the final global model...")
                
                # Create a DecisionTreeClassifier and store parameters
                model = DecisionTreeClassifier()
                with open("global_model.pkl", "wb") as f:
                    pickle.dump(model, f)
                
                print("✅ Model saved as 'global_model.pkl'!")

        return aggregated_parameters, _

def main():
    """Start the Flower server with a custom strategy to save the model."""
    strategy = CustomStrategy()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
