import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

class GameReader():
    def __init__(self,
               path: str = 'data\\kaggle\\march-machine-learning-mania-2025\\MRegularSeasonCompactResults.csv',
               name_path: str = 'data\\kaggle\\march-machine-learning-mania-2025\\MTeams.csv'):
        """Should read in the history of games across all teams"""
        self.df = pd.read_csv(path)
        name_df = pd.read_csv(name_path)
        self.names = {row['TeamName']: row['TeamID'] for _, row in name_df.iterrows()}

    def games_bt(self, team1: str, team2: str) -> pd.DataFrame:
        """Return all games played between two teams by name

        Args:
            team1 (str): first team name
            team2 (str): second team name

        Returns:
            pd.DataFrame: subset of games played between teams
        """
        id1 = self.names[team1]
        id2 = self.names[team2]
        team1_wins = self.df.loc[(self.df['WTeamID']==id1) & (self.df['LTeamID']==id2),:]
        team2_wins = self.df.loc[(self.df['WTeamID']==id2) & (self.df['LTeamID']==id1), :]

        team1_wins = team1_wins.rename(columns={'WTeamID':'Team1ID',
                                        'WScore':'Team1Score',
                                        'LTeamID': 'Team2ID',
                                        'LScore': 'Team2Score',
                                        'WLoc':'Team1Home'})
        
        team1_wins['Team1Home'] = team1_wins['Team1Home'].map({'N':0,'H':1,'A':-1})
        
        team2_wins = team2_wins.rename(columns={'WTeamID':'Team2ID',
                                        'WScore':'Team2Score',
                                        'LTeamID': 'Team1ID',
                                        'LScore': 'Team1Score',
                                        'WLoc':'Team1Home'})
        
        team2_wins['Team1Home'] = team2_wins['Team1Home'].map({'N':0,'H':-1,'A':1})
        

        return pd.concat([team1_wins, team2_wins]).reset_index(drop=True)
    
    def flatten_all_games(self)-> pd.DataFrame:
        """Flattens games into one row per team/game"""
        df = self.df.copy()

        # Wins
        win_df = df.rename(columns={
            'WTeamID': 'TeamID', 'WScore': 'Score',
            'LTeamID': 'OppID', 'LScore': 'OppScore',
            'WLoc': 'Loc'
        })
        win_df['Win'] = 1

        # Losses
        lose_df = df.rename(columns={
            'LTeamID': 'TeamID', 'LScore': 'Score',
            'WTeamID': 'OppID', 'WScore': 'OppScore',
        })
        lose_df['Loc'] = lose_df['WLoc'].map({'H': -1, 'A': 1, 'N': 0})
        lose_df = lose_df.drop(columns=['WLoc'])
        lose_df['Win'] = 0

        all_games = pd.concat([win_df, lose_df], ignore_index=True)
        all_games['Margin'] = all_games['Score'] - all_games['OppScore']
        all_games['Score'] = pd.to_numeric(all_games['Score'], errors='coerce')
        all_games['OppScore'] = pd.to_numeric(all_games['OppScore'], errors='coerce')
        all_games['Win'] = pd.to_numeric(all_games['Win'], errors='coerce')
        all_games['Margin'] = pd.to_numeric(all_games['Margin'], errors='coerce')
        all_games['Loc'] = pd.to_numeric(all_games['Loc'], errors='coerce')
        all_games = all_games.dropna(subset=['Score', 'OppScore', 'Win', 'Margin', 'Loc'])


        return all_games

    def team_season_stats(self) -> pd.DataFrame:
        """Return season-level stats for each team."""
        games = self.flatten_all_games()
        team_stats = games.groupby(['Season', 'TeamID']).agg({
            'Score': 'mean',
            'OppScore': 'mean',
            'Margin': 'mean',
            'Win': 'mean',
            'Loc': 'mean'
        }).rename(columns={
            'Score': 'AvgScore',
            'OppScore': 'AvgOppScore',
            'Margin': 'AvgMargin',
            'Win': 'WinRate',
            'Loc': 'AvgHomeIndicator'  # (1,-1) -> (home, away) scale
        }).reset_index()

        return team_stats

    def train_xgboost_model(self, tourney_df: pd.DataFrame):
        """Train an XGBoost model on historical tournament matchups."""
        self.model = None

        # Step 1: flatten tournament results
        df1 = tourney_df[['Season', 'WTeamID', 'WScore', 'LTeamID', 'LScore']].copy()
        df1.columns = ['Season', 'Team1ID', 'Team1Score', 'Team2ID', 'Team2Score']
        df1['Label'] = 1

        df2 = tourney_df[['Season', 'WTeamID', 'WScore', 'LTeamID', 'LScore']].copy()
        df2.columns = ['Season', 'Team2ID', 'Team2Score', 'Team1ID', 'Team1Score']
        df2['Label'] = 0

        matchup_df = pd.concat([df1, df2]).reset_index(drop=True)

        # Step 2: get season stats
        team_stats = self.team_season_stats()

        # Merge in stats
        merged = matchup_df.merge(team_stats, left_on=['Season', 'Team1ID'], right_on=['Season', 'TeamID'], how='left') \
                           .rename(columns=lambda x: 'T1_' + x if x in ['AvgScore', 'AvgOppScore', 'AvgMargin', 'WinRate'] else x) \
                           .drop(columns=['TeamID'])

        merged = merged.merge(team_stats, left_on=['Season', 'Team2ID'], right_on=['Season', 'TeamID'], how='left') \
                       .rename(columns=lambda x: 'T2_' + x if x in ['AvgScore', 'AvgOppScore', 'AvgMargin', 'WinRate'] else x) \
                       .drop(columns=['TeamID'])

        # Step 3: feature differences
        merged['Diff_AvgScore'] = merged['T1_AvgScore'] - merged['T2_AvgScore']
        merged['Diff_AvgOppScore'] = merged['T1_AvgOppScore'] - merged['T2_AvgOppScore']
        merged['Diff_AvgMargin'] = merged['T1_AvgMargin'] - merged['T2_AvgMargin']
        merged['Diff_WinRate'] = merged['T1_WinRate'] - merged['T2_WinRate']

        # Step 4: Train XGBoost
        features = ['Diff_AvgScore', 'Diff_AvgOppScore', 'Diff_AvgMargin', 'Diff_WinRate']
        X = merged[features]
        y = merged['Label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=60, max_depth=5, verbosity=0)
        model.fit(X_train, y_train)

        self.model = model  # store model for future use

        print("Training complete.")
        print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")
        print(f"Log Loss: {log_loss(y_test, model.predict_proba(X_test)[:, 1]):.3f}")

     def train_logreg(self, tourney_df):
        """Train a Logistic Regression model on historical tournament matchups."""
        self.model = None

        # Step 1: flatten tournament results
        df1 = tourney_df[['Season', 'WTeamID', 'WScore', 'LTeamID', 'LScore']].copy()
        df1.columns = ['Season', 'Team1ID', 'Team1Score', 'Team2ID', 'Team2Score']
        df1['Label'] = 1

        df2 = tourney_df[['Season', 'WTeamID', 'WScore', 'LTeamID', 'LScore']].copy()
        df2.columns = ['Season', 'Team2ID', 'Team2Score', 'Team1ID', 'Team1Score']
        df2['Label'] = 0

        matchup_df = pd.concat([df1, df2]).reset_index(drop=True)

        # Step 2: get season stats
        team_stats = self.team_season_stats()

        # Merge in stats
        merged = matchup_df.merge(team_stats, left_on=['Season', 'Team1ID'], right_on=['Season', 'TeamID'], how='left') \
                           .rename(columns=lambda x: 'T1_' + x if x in ['AvgScore', 'AvgOppScore', 'AvgMargin', 'WinRate'] else x) \
                           .drop(columns=['TeamID'])

        merged = merged.merge(team_stats, left_on=['Season', 'Team2ID'], right_on=['Season', 'TeamID'], how='left') \
                       .rename(columns=lambda x: 'T2_' + x if x in ['AvgScore', 'AvgOppScore', 'AvgMargin', 'WinRate'] else x) \
                       .drop(columns=['TeamID'])

        # Step 3: feature differences
        merged['Diff_AvgScore'] = merged['T1_AvgScore'] - merged['T2_AvgScore']
        merged['Diff_AvgOppScore'] = merged['T1_AvgOppScore'] - merged['T2_AvgOppScore']
        merged['Diff_AvgMargin'] = merged['T1_AvgMargin'] - merged['T2_AvgMargin']
        merged['Diff_WinRate'] = merged['T1_WinRate'] - merged['T2_WinRate']

        # Step 4: Train Logistic Regression
        features = ['Diff_AvgScore', 'Diff_AvgOppScore', 'Diff_AvgMargin', 'Diff_WinRate']
        X = merged[features]
        y = merged['Label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        model = LogisticRegression(solver='lbfgs', max_iter=1000)
        model.fit(X_train, y_train)

        self.model = model  # store model for future use

        print("Logistic Regression training complete.")
        print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")
        print(f"Log Loss: {log_loss(y_test, model.predict_proba(X_test)[:, 1]):.3f}")

    
    def predict_matchup(self, team1_name: str, team2_name: str, season: int) -> float:
        """Predict the probability that team1 beats team2 using the trained model."""
        id1 = self.names[team1_name]
        id2 = self.names[team2_name]

        # Get season stats
        stats = self.team_season_stats()
        row1 = stats[(stats['Season'] == season) & (stats['TeamID'] == id1)]
        row2 = stats[(stats['Season'] == season) & (stats['TeamID'] == id2)]

        if row1.empty or row2.empty:
            raise ValueError("Season stats not found for one or both teams.")

        # Extract values
        diff = {'Diff_AvgScore': row1['AvgScore'].values[0] - row2['AvgScore'].values[0],
                'Diff_AvgOppScore': row1['AvgOppScore'].values[0] - row2['AvgOppScore'].values[0],
                'Diff_AvgMargin': row1['AvgMargin'].values[0] - row2['AvgMargin'].values[0],
                'Diff_WinRate': row1['WinRate'].values[0] - row2['WinRate'].values[0]}

        X = pd.DataFrame([diff])
        prob = self.model.predict_proba(X)[0, 1]  # probability Team1 wins
        return prob

if __name__ == '__main__':
    gr = GameReader()
    #print(gr.games_bt('Duquesne', 'Bowling Green'))
    #print(gr.team_season_stats().head())
    tourney_df = pd.read_csv('data\\kaggle\\march-machine-learning-mania-2025\\MNCAATourneyCompactResults.csv')
    gr.train_xgboost_model(tourney_df)
    gr.predict_matchup('BYU', 'Duquesne', 2022)
    prob = gr.predict_matchup('BYU', 'Duquesne', 2022)
    print(f"Probability Team1 beats Team2: {prob:.2%}")
