import pandas as pd

# Load CSV
df = pd.read_csv("combined_fighter_data.csv")

# Remove '%' and convert to float
percentage_columns = ['win_rate', 'loss_rate', 'draw_rate', 'dc_nc_rate']
for col in percentage_columns:
    df[col] = df[col].str.replace('%', '').astype(float) / 100  # Convert to decimal

# Save cleaned CSV
df.to_csv("cleaned_fighter_data.csv", index=False)
