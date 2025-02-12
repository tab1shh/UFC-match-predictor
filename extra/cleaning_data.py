import pandas as pd

fighter_data = pd.read_csv('combined_fighter_data.csv', encoding='latin-1')

# calculate medians
median_height_by_weight = fighter_data.groupby('weight_pounds')['height_cm'].median().to_dict()
median_weight_by_height = fighter_data.groupby('height_cm')['weight_pounds'].median().to_dict()
median_reach_by_height = fighter_data.groupby('height_cm')['reach_cm'].median().to_dict()
median_reach_by_weight = fighter_data.groupby('weight_pounds')['reach_cm'].median().to_dict()

# function replace NA in height with median height based on weight
def fill_height(row, median_height_by_weight):
    if pd.isna(row['height_cm']):
        return median_height_by_weight.get(row['weight_pounds'], row['height_cm'])
    return row['height_cm']

# function to replace NA in weight with median weight based on height
def fill_weight(row, median_weight_by_height):
    if pd.isna(row['weight_pounds']):
        return median_weight_by_height.get(row['height_cm'], row['weight_pounds'])
    return row['weight_pounds']

# function to replace NA in reach with median reach based on height or weight
def fill_reach(row, median_reach_by_height, median_reach_by_weight):
    if pd.isna(row['reach_cm']):
        if not pd.isna(row['height_cm']):
            return median_reach_by_height.get(row['height_cm'], row['reach_cm'])
        elif not pd.isna(row['weight_pounds']):
            return median_reach_by_weight.get(row['weight_pounds'], row['reach_cm'])
    return row['reach_cm']


# apply the functions to fill missing values
fighter_data['height_cm'] = fighter_data.apply(fill_height, axis=1, median_height_by_weight=median_height_by_weight)
fighter_data['weight_pounds'] = fighter_data.apply(fill_weight, axis=1, median_weight_by_height=median_weight_by_height)
fighter_data['reach_cm'] = fighter_data.apply(fill_reach, axis=1, median_reach_by_height=median_reach_by_height, median_reach_by_weight=median_reach_by_weight)

# drop rows where both height and weight are NA,
fighter_data.dropna(subset=['height_cm', 'weight_pounds'], how='all', inplace=True)

fighter_data = fighter_data.dropna(subset=['date_of_birth'])
most_common_stance = fighter_data['stance'].mode()[0]
fighter_data['stance'] = fighter_data['stance'].fillna(most_common_stance)

'''
want to remove all inactive fighters from the cleaned data as well
so in the final there should be around 900 fighters
'''

# Save the cleaned data
fighter_data.to_csv('extra/cleaned_fighters.csv', index=False)
print("Successfully cleaned")