import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

start_time = time.time()

# Load data
file_path = 'cleaned_fighters.csv'
ufc_stats = pd.read_csv(file_path, encoding='latin-1')

# Encode categorical columns
le = LabelEncoder()
ufc_stats['stance'] = le.fit_transform(ufc_stats['stance'])

# Convert percentages to numerical data
for col in ['win_rate', 'loss_rate', 'draw_rate', 'dc_nc_rate', 'career_StrDef', 'career_TD_Acc', 'career_TD_Def']:
    ufc_stats[col] = ufc_stats[col].str.replace('%', '').astype(float) / 100

def get_fighter_stats(name, df):
    """Fetch fighter stats from the DataFrame."""
    return df[df['fighter_name'] == name].iloc[0]

def compute_features(f1, f2, df):
    """Find numerical differences between two fighters' stats."""
    f1_stats = get_fighter_stats(f1, df)
    f2_stats = get_fighter_stats(f2, df)
    features = {}
    for col in df.columns:
        if df[col].dtype in [float, int] and col != 'fighter_name':
            features[f"{col}_diff"] = abs(f1_stats[col] - f2_stats[col])
    return features

# Prepare data for training
features = []
labels = []

for i in range(len(ufc_stats)):
    for j in range(i + 1, len(ufc_stats)):
        f1 = ufc_stats.iloc[i]['fighter_name']
        f2 = ufc_stats.iloc[j]['fighter_name']
        computing_features = compute_features(f1, f2, ufc_stats)
        features.append(list(computing_features.values()))
        labels.append(1 if ufc_stats.iloc[i]['f_wins'] > ufc_stats.iloc[j]['f_wins'] else 0)

# Convert to DataFrame
features = pd.DataFrame(features)
labels = pd.Series(labels)

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

end_time = time.time()

print(f"Training Time: {end_time - start_time:.2f} seconds")