import pandas as pd
import numpy as np
import joblib 
from sklearn.preprocessing import LabelEncoder, StandardScaler

model = joblib.load('fighter_predictor_model.pkl')

file_path = 'combined_fighter_data.csv'
new_data = pd.read_csv(file_path, encoding='latin-1')

# Add a default value for 'career_KD_Avg'
new_data['career_KD_Avg'] = 0.0  # or any other default value

new_data['stance'] = new_data['stance'].astype(str)

# Encode categorical columns
le = LabelEncoder()
new_data['stance'] = le.fit_transform(new_data['stance'])

# Convert percentages to numerical data
for col in ['win_rate', 'loss_rate', 'draw_rate', 'dc_nc_rate', 'career_StrDef', 'career_TD_Acc', 'career_TD_Def']:
    new_data[col] = new_data[col].str.replace('%', '').astype(float) / 100

# Convert fighter names to lowercase for case-insensitive comparison
fighter_stats = {row['fighter_name'].lower(): row for _, row in new_data.iterrows()}

def compute_features(f1, f2):
    """Find numerical differences between two fighters' stats."""
    f1_stats = fighter_stats[f1.lower()]
    f2_stats = fighter_stats[f2.lower()]
    features = {}
    for col in new_data.columns:
        if new_data[col].dtype in [float, int] and col != 'fighter_name':
            features[f"{col}_diff"] = abs(f1_stats[col] - f2_stats[col])
    
    return features 

fighter_1 = input("Enter first fighter's name: ").strip()
fighter_2 = input("Enter second fighter's name: ").strip()

if fighter_1.lower() in fighter_stats and fighter_2.lower() in fighter_stats:
    match_features = compute_features(fighter_1, fighter_2)
    match_features_df = pd.DataFrame([match_features])

    # Debugging: Print the features being generated
    # print("Generated Features:")
    # print(match_features_df.columns.tolist())

    # Scale the features (Use the same scaler from training)
    scaler = StandardScaler()
    match_features_scaled = scaler.fit_transform(match_features_df)

    # Predict match outcome
    prediction = model.predict(match_features_scaled)[0]

    # Interpret the result
    winner = fighter_1 if prediction == 1 else fighter_2
    print(f"\nPredicted Winner: {winner}")
else:
    print("\nOne or both fighters not found in the dataset. Please check the spelling or try another pair.")