import pandas as pd

# Load CSV
df = pd.read_csv("combined_fighter_data.csv")
df2 = pd.read_csv("filtered_fighters_data.csv")

# Identify all columns that contain percentages
percentage_columns = [
    'win_rate', 'loss_rate', 'draw_rate', 'dc_nc_rate',
    'career_StrAcc', 'career_StrDef', 'career_TD_Avg',
    'career_TD_Acc', 'career_TD_Def'
]

# Remove '%' and convert to float (divide by 100 if percentages should be decimals)
for col in percentage_columns:
    df[col] = df[col].astype(str).str.replace('%', '', regex=False)  # Remove '%'
    df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to float (NaN if invalid)

for col in percentage_columns:
    df2[col] = df2[col].astype(str).str.replace('%', '', regex=False)  # Remove '%'
    df2[col] = pd.to_numeric(df2[col], errors='coerce')  # Convert to float (NaN if invalid)

# Save cleaned CSV
df.to_csv("cleaned_fighter_data.csv", index=False)
df2.to_csv("predictor_data.csv", index=False)

print("CSV cleaned successfully!")
