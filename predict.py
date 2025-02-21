import joblib
import pandas as pd
import pickle

# Load the trained model
model = joblib.load("trained_model.pkl")

# Load feature names (to ensure input data matches model training)
with open("trained_model.pkl", "rb") as f:
    feature_names = pickle.load(f)

def get_user_input():
    """Get symptoms from user as input flags."""
    print("\n💡 Enter symptoms as comma-separated values (e.g., fever,cough).")
    print("🔍 Available Symptoms:", ", ".join(feature_names))
    
    user_input = input("\n👉 Enter your symptoms: ").strip().lower().split(",")

    # Create a dictionary with all symptoms set to 0
    symptoms_dict = {feature: 0 for feature in feature_names}

    # Update symptoms present in the user's input
    for symptom in user_input:
        symptom = symptom.strip()
        if symptom in symptoms_dict:
            symptoms_dict[symptom] = 1  # Set present symptoms to 1

    return pd.DataFrame([symptoms_dict])  # Convert to DataFrame

def predict_disease():
    """Predict disease based on user symptoms."""
    symptoms_df = get_user_input()
    prediction = model.predict(symptoms_df)
    print("\n🩺 Predicted Disease:", prediction[0])

# Run interactive prediction
if __name__ == "__main__":
    print("\n💡 Welcome to the AI Disease Predictor!")
    predict_disease()
