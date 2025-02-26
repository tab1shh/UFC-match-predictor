'''
want to go through the combined_fighter_data.csv file and go through the ufc_rankings.csv file and filter out such that
only fighters from ufc_rankings are taken from the combined_fighter_data.csv with there stats too. 
So take their name, stats, etc
'''
import pandas as pd

combined_fighter = pd.read_csv('csv_files/combined_fighter_data.csv')
ufc_rankings = pd.read_csv('csv_files/ufc_rankings.csv')

combined_fighter = combined_fighter.drop_duplicates(subset=['fighter_name'], keep='first')
ufc_rankings = ufc_rankings.drop_duplicates(subset=['Name'], keep='first')

merged_fighters = pd.merge(combined_fighter, ufc_rankings, left_on='fighter_name', right_on='Name', how='inner')

merged_fighters.drop(columns=['Name'], inplace=True)

merged_fighters.to_csv('csv_files/predictor_data.csv', index=False)
print(merged_fighters)
