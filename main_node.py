import flwr as fl
import pickle

class CustomStrategy(fl.server.strategy.FedAvg):
    """Custom strategy to save the global model after training rounds."""

    def aggregate_fit(self, rnd, results, failures):
        """Aggregates model updates from clients and saves the final model."""
        aggregated_parameters, _ = super().aggregate_fit(rnd, results, failures)

        if aggregated_parameters is not None:
            print(f"✅ Round {rnd} - Model Aggregated Successfully!")

            # Save the model after the final training round
            if rnd == 3:  # Change this if you modify num_rounds
                print("💾 Saving the final global model...")
                with open("global_model.pkl", "wb") as f:
                    pickle.dump(aggregated_parameters, f)
                print("✅ Model saved as 'global_model.pkl'!")

        return aggregated_parameters, _

def main():
    """Starts the Flower server with a custom strategy to save the model."""
    strategy = CustomStrategy()  # ✅ Removed unexpected 'num_rounds' argument

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy  # ✅ Added strategy for model saving
    )

if __name__ == "__main__":  # ✅ Fixed the naming issue
    main()
