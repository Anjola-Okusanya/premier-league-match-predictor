import pandas as pd
pd.set_option('display.max_columns', None)

#Load in raw Premier League match data
df = pd.read_csv('data/E0.csv') 

#Function returns a list of ALL matches a specific team played with stats like Results, Form etc
def team_stats(df, team_name):

    # Filter for when the team is playing at Home
    home_filt = df['HomeTeam'] == team_name
    team_home = df[home_filt].copy()

    # Adding columns to the data frame: Stores team name, The opponent (Away Team), Goals scored and conceded by the team when at home and if they are at home
    team_home['Team'] = team_name
    team_home['Opponent'] = team_home['AwayTeam']
    team_home['GoalsScored'] = team_home['FTHG']
    team_home['GoalsConceded'] = team_home['FTAG']
    team_home['HomeAway'] = 'Home'
    
    # Repeated process for when the team is Away from home
    away_filt = df['AwayTeam'] == team_name
    team_away = df[away_filt].copy()
    team_away['Team'] = team_name
    team_away['Opponent'] = team_away['HomeTeam']
    team_away['GoalsScored'] = team_away['FTAG']
    team_away['GoalsConceded'] = team_away['FTHG']
    team_away['HomeAway'] = 'Away'

    # Combine and stack home and away matches vertically
    team_all_games = pd.concat([team_home, team_away])

    # Converts Date column from a string into a datetime object (Allows for rolling windows - Form)
    team_all_games['Date'] = pd.to_datetime(team_all_games['Date'], format='%d/%m/%Y') 

    # Keeps only relevant columns after sorting the matches chronologically
    team_all_games = team_all_games.sort_values('Date').reset_index(drop=True) # Sorts dataframe by date
    team_all_games = team_all_games[['Date', 'Team', 'Opponent', 'GoalsScored', 'GoalsConceded', 'HomeAway']]

    # Determines the result of a game
    def result (goals_scored, goals_conceded):
        if goals_scored > goals_conceded:
            return 'W'
        elif goals_scored == goals_conceded:
            return 'D'
        else:
            return 'L'
        
    # Applying function by creating result column and using lambda
    team_all_games['Result'] = team_all_games.apply(
        lambda row: result(row['GoalsScored'], row['GoalsConceded']),
        axis=1
    )
    # Map the results to league points (3pts win, 1 draw, 0 loss)
    team_all_games['Points'] = team_all_games['Result'].map({'W': 3, 'D': 1, 'L': 0})
    
    # Form calculated by rolling sum of points over the last 5 matches
    team_all_games['Form'] = team_all_games['Points'].rolling(window=5).sum()

    # Features using a rolling window of 5 games to approx attacking and defensive strength
    team_all_games['GoalsScoredLast5'] = team_all_games['GoalsScored'].rolling(5).sum()
    team_all_games['GoalsConcededLast5'] = team_all_games['GoalsConceded'].rolling(5).sum()
    team_all_games['WinsLast5'] = (team_all_games['Result'] == 'W').rolling(5).sum()
    print(team_all_games[['Date', 'Team', 'Opponent', 'GoalsScored', 'GoalsConceded', 'Result', 'Points','Form', 'GoalsScoredLast5', 'GoalsConcededLast5', 'WinsLast5']])

    return team_all_games

all_team_names = df['HomeTeam'].unique()
all_team_data = []

# Make rolling stats for each team and append to list
for team in all_team_names:
    answer = team_stats(df, team)
    all_team_data.append(answer)

# Combine into one DataFrame
full_dataset = pd.concat(all_team_data, ignore_index=True)

print(f"Total rows: {len(full_dataset)}")
print(f"Unique teams: {full_dataset['Team'].nunique()}")
full_dataset.to_csv('team_rolling_stats.csv', index=False)

