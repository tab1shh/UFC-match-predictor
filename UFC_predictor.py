import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

start_time = time.time()

file_path = 'cleaned_fighters.csv'
ufc_stats = pd.read_csv(file_path, encoding='latin-1')

# encode categorical columns
le = LabelEncoder()
ufc_stats['stance'] = le.fit_transform(ufc_stats['stance'])

# convert percentages to numerical data
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
    # abs difference
    for col in df.columns:
        if df[col].dtype in [float, int] and col != 'fighter_name':
            features[f"{col}_diff"] = abs(f1_stats[col] - f2_stats[col])

    # ratio between two fighters' stats
    for col in df.columns:
        if df[col].dtype in [float, int] and col != 'fighter_name' and f2_stats[col] != 0:
            features[f"{col}_ratio"] = f1_stats[col] / f2_stats[col]
    
    # interaction features like str_acc and such
    for col1 in df.columns:
        for col2 in df.columns:
            if df[col1].dtype in [float, int] and df[col2].dtype in [float, int] and col1 != 'fighter_name' and col2 != 'fighter_name':
                features[f"{col1}_{col2}_interaction"] = f1_stats[col1] * f2_stats[col2]

    return features


# prepare data for training
features = []
labels = []

for i in range(len(ufc_stats)):
    for j in range(i + 1, len(ufc_stats)):
        f1 = ufc_stats.iloc[i]['fighter_name']
        f2 = ufc_stats.iloc[j]['fighter_name']
        computing_features = compute_features(f1, f2, ufc_stats)
        features.append(list(computing_features.values()))
        labels.append(1 if ufc_stats.iloc[i]['f_wins'] > ufc_stats.iloc[j]['f_wins'] else 0)

features = pd.DataFrame(features)
labels = pd.Series(labels)

# train-test split
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# hyperparameter Tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring='accuracy'
)
grid_search.fit(features_train, labels_train)

# best model from GridSearchCV
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# evaluate the model
labels_pred = best_model.predict(features_test)
print("Accuracy:", accuracy_score(labels_test, labels_pred))

end_time = time.time()  

print(f"Training Time: {end_time - start_time:.2f} seconds")

import joblib

# save the model to a file
joblib.dump(best_model, 'fighter_predictor_model.pkl')
print("Model saved successfully!")
