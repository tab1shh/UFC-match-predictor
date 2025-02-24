import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time
import joblib
from joblib import Parallel, delayed

start_time = time.time()

# Load data
file_path = 'predictor_data.csv'
ufc_stats = pd.read_csv(file_path, encoding='latin-1')

# handle missing values
ufc_stats = ufc_stats.fillna(0)

# Encode categorical columns
le = LabelEncoder()
ufc_stats['stance'] = le.fit_transform(ufc_stats['stance'])

# Convert percentages to numerical data
for col in ['win_rate', 'loss_rate', 'draw_rate', 'dc_nc_rate', 'career_StrDef', 'career_TD_Acc', 'career_TD_Def']:
    ufc_stats[col] = ufc_stats[col].astype(str).str.replace('%', '').astype(float) / 100

fighter_stats = {row['fighter_name']: row for _, row in ufc_stats.iterrows()}

def compute_features(f1, f2):
    """Find numerical differences between two fighters' stats."""
    f1_stats = fighter_stats[f1]
    f2_stats = fighter_stats[f2]
    features = {}
    for col in ufc_stats.columns:
        if ufc_stats[col].dtype in [float, int] and col != 'fighter_name':
            features[f"{col}_diff"] = abs(f1_stats[col] - f2_stats[col])
    return features 

# Prepare data for training
fighters = ufc_stats['fighter_name'].tolist()

def process_pair(i, j):
    f1 = fighters[i]
    f2 = fighters[j]
    computing_features = compute_features(f1, f2)
    label = 1 if fighter_stats[f1]['f_wins'] > fighter_stats[f2]['f_wins'] else 0
    return list(computing_features.values()), label

results = Parallel(n_jobs=-1)(delayed(process_pair)(i, j) for i in range(len(fighters)) for j in range(i + 1, len(fighters)))

# Unpack results into features and labels
features, labels = zip(*results)
features = pd.DataFrame(features)
labels = pd.Series(labels)

# features = []
# labels = []

# fighters = ufc_stats['fighter_name'].tolist()
# for i in range(len(fighters)):
#     for j in range(i + 1, len(fighters)):
#         f1 = fighters[i]
#         f2 = fighters[j]
#         computing_features = compute_features(f1, f2)
#         features.append(list(computing_features.values()))
#         labels.append(1 if fighter_stats[i]['f_wins'] > fighter_stats[j]['f_wins'] else 0)

# # Convert to DataFrame
# features = pd.DataFrame(features)
# labels = pd.Series(labels)

scaler = StandardScaler()
features = scaler.fit_transform(features)

# Train-test split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)
# Train the model
model = RandomForestClassifier()
model.fit(features_train, labels_train)

# Evaluate the model
labels_pred = model.predict(features_test)
print("Accuracy:", accuracy_score(labels_test, labels_pred))

cv_scores = cross_val_score(model, features, labels, cv=5)
print("Cross-Validation Accuracy:", np.mean(cv_scores))

end_time = time.time()  

print(f"Training Time: {end_time - start_time:.2f} seconds")

# save the model to a file
joblib.dump(model, 'fighter_predictor_model.pkl')
print("Model saved successfully!")