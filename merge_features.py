import pandas as pd

# Load in original Premier League match data (now match based with a row per match)
df = pd.read_csv('data/E0.csv')

# Convert Date column from string to datetime objects
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Loadin stats for each team and convert date column here aswell
team_stats = pd.read_csv('team_rolling_stats.csv')
team_stats['Date'] = pd.to_datetime(team_stats['Date'])

# Create dataframe with one row per maych
matches = df[['Date', 'HomeTeam', 'AwayTeam', 'FTR']].copy()

# Merge the home teams rolling stats into the matches dataframe
matches = matches.merge(
    team_stats,
    left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left'
)

#Merges the away teams rolling stats into the matches dataframe
matches = matches.merge(
    team_stats,
    left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left'
)

# Select and rename only the columns needed for Machine Learning
matches = matches[['Date','HomeTeam', 'AwayTeam', 'Form_x', 'Form_y', 'GoalsScoredLast5_x', 'GoalsScoredLast5_y', 'Result_x', 'WinsLast5_x', 'WinsLast5_y', 'GoalsConcededLast5_x', 'GoalsConcededLast5_y']]
matches.rename(columns={'Form_x': 'HomeForm', 'Form_y': 'AwayForm', 'GoalsScoredLast5_x': 'HomeGoalsLast5', 'GoalsScoredLast5_y': 'AwayGoalsLast5', 'Result_x': 'Result', 'WinsLast5_x': 'HomeWinsLast5', 'WinsLast5_y': 'AwayWinsLast5', 'GoalsConcededLast5_x':'HomeGoalsConcededLast5', 'GoalsConcededLast5_y': 'AwayGoalsConcededLast5'}, inplace=True)

# Sanity checks, print final feature table and save
print(f"Matches: {len(matches)} rows")
pd.set_option('display.max_columns', None)
print(matches)
matches.to_csv('match_features.csv', index=False)
