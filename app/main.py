import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import difflib
import traceback

SEQ_LENGTH = 5

def format_name(name):
    parts = name.strip().split()
    if len(parts) == 1:
        return parts[0]
    initials = " ".join(p[0] for p in parts[:-1])
    return f"{initials} {parts[-1]}"

def preprocess_data(df1, df2=None, seq_length=SEQ_LENGTH):
    df1 = df1.dropna()
    df = pd.concat([df1, df2.dropna()]) if df2 is not None else df1

    required_features = [
        'Total_Runs', 'Balls_Faced', 'Times_Out', '4s', '6s',
        'Strike_Rate', 'Wickets', 'Balls_Bowled', 'Economy',
        'Fantasy_Batting', 'Fantasy_Bowling', 'Total_Fantasy_Points', 'Player'
    ]

    for feat in required_features:
        if feat not in df.columns:
            raise KeyError(f"Missing feature: {feat}")

    for feat in required_features[:-2] + ['Total_Fantasy_Points']:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')
    df = df.dropna(subset=required_features)

    sequences, targets, players = [], [], []
    for _, row in df.iterrows():
        seq = np.array([[
            row['Total_Runs'] * (i+1)/seq_length,
            row['Balls_Faced'] * (i+1)/seq_length,
            row['Times_Out'] * (i+1)/seq_length,
            row['4s'] * (i+1)/seq_length,
            row['6s'] * (i+1)/seq_length,
            row['Strike_Rate'],
            row['Wickets'] * (i+1)/seq_length,
            row['Balls_Bowled'] * (i+1)/seq_length,
            row['Economy'],
            row['Fantasy_Batting'],
            row['Fantasy_Bowling']
        ] for i in range(seq_length)])
        sequences.append(seq)
        targets.append(row['Total_Fantasy_Points'])
        players.append(row['Player'])

    return np.array(sequences), np.array(targets), players

def create_nn_model(input_shape):
    # Flatten the sequences for MLPRegressor
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    return model

class FantasyTeamSelector:
    def __init__(self, model):
       self.model = model

    def assign_manual_roles(self, df, manual_roles_dict):
        df = df.copy()
        df['Player_Clean'] = df['Player'].str.strip().str.lower()
        df['Role'] = df['Player_Clean'].map(manual_roles_dict).fillna('Batsman')
        return df

    def select_team(self, players_df, seqs, num_players=11):
        # Flatten sequences for prediction
        seqs_flat = seqs.reshape(seqs.shape[0], -1)
        preds = self.model.predict(seqs_flat).flatten()
        players_df = players_df.copy()
        players_df['Predicted_Score'] = preds

        role_requirements = {'Wicketkeeper': 1, 'Batsman': 3, 'Allrounder': 3, 'Bowler': 4}
        selected = []

        for role, count in role_requirements.items():
            role_players = players_df[players_df['Role'] == role]
            top_players = role_players.nlargest(count, 'Predicted_Score')
            selected.append(top_players)
            players_df = players_df.drop(top_players.index)

        team = pd.concat(selected)
        if len(team) < num_players:
            extra = players_df.nlargest(num_players - len(team), 'Predicted_Score')
            team = pd.concat([team, extra])

        team = team.nlargest(num_players, 'Predicted_Score')
        captain, vice_captain = team.iloc[0], team.iloc[1]
        return team, captain, vice_captain


# --------------------- Streamlit UI ---------------------

st.title("🏏 Fantasy Cricket Team Selector")

csv1 = st.file_uploader("Upload Main Dataset CSV (with fantasy points)", type="csv")
csv2 = st.file_uploader("Upload Additional Dataset CSV (optional)", type="csv")

if csv1:
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2) if csv2 else None

    try:
        seqs, targets, players = preprocess_data(df1, df2)
        
        # Flatten sequences for MLPRegressor
        seqs_flat = seqs.reshape(seqs.shape[0], -1)
        
        model = create_nn_model(seqs.shape)
        model.fit(seqs_flat, targets)

        players_df = pd.DataFrame({
            'Player': players,
            'Total_Runs': seqs[:, -1, 0],
            'Balls_Faced': seqs[:, -1, 1],
            'Times_Out':   seqs[:, -1, 2],
            '4s':          seqs[:, -1, 3],
            '6s':          seqs[:, -1, 4],
            'Strike_Rate': seqs[:, -1, 5],
            'Wickets':     seqs[:, -1, 6],
            'Balls_Bowled':seqs[:, -1, 7],
            'Economy':     seqs[:, -1, 8],
            'Fantasy_Batting': seqs[:, -1, 9],
            'Fantasy_Bowling': seqs[:, -1, 10]
        })
        players_df['Player_Clean'] = players_df['Player'].str.strip().str.lower()

        players_with_roles = [
            ("Quinton de Kock", "Wicketkeeper"), ("Moeen Ali", "Allrounder"),
            ("Angkrish Raghuvanshi", "Batsman"), ("Ajinkya Rahane", "Batsman"),
            ("Venkatesh Iyer", "Batsman"), ("Ramandeep Singh", "Batsman"),
            ("Andre Russell", "Allrounder"), ("Sunil Narine", "Allrounder"),
            ("Varun Chakravarthy", "Bowler"), ("Harshit Rana", "Bowler"),
            ("Anrich Nortje", "Bowler"), ("Jos Buttler", "Wicketkeeper"),
            ("Shubman Gill", "Batsman"), ("Sai Sudharsan", "Batsman"),
            ("Sherfane Rutherford", "Batsman"), ("Shahrukh Khan", "Allrounder"),
            ("Rahul Tewatia", "Allrounder"), ("R Sai Kishore", "Bowler"),
            ("Arshad Khan", "Bowler"), ("Rashid Khan", "Bowler"),
            ("Prasidh Krishna", "Bowler"), ("Mohammed Siraj", "Bowler")
        ]
        manual_roles_dict = {format_name(name).lower(): role for name, role in players_with_roles}

        given_players = list(manual_roles_dict.keys())
        dataset_players = players_df['Player_Clean'].unique().tolist()
        name_map = {}
        for player in given_players:
            match = difflib.get_close_matches(player, dataset_players, n=1, cutoff=0.7)
            if match:
                name_map[player] = match[0]

        mapped_names = list(name_map.values())
        filtered_indices = players_df.index[players_df['Player_Clean'].isin(mapped_names)]
        filtered_df = players_df.loc[filtered_indices]
        filtered_seqs = seqs[filtered_indices]

        selector = FantasyTeamSelector(model)
        filtered_df = selector.assign_manual_roles(filtered_df, manual_roles_dict)

        if len(filtered_df) < 11:
            st.error("Not enough valid players found for team selection.")
        else:
            team, captain, vice_captain = selector.select_team(filtered_df, filtered_seqs)
            st.subheader("Optimal Fantasy 11")
            st.dataframe(team[['Player', 'Role', 'Predicted_Score']])

            st.success(f"⭐ Captain: {captain['Player']} (Score: {captain['Predicted_Score']:.2f})")
            st.info(f"⭐ Vice-Captain: {vice_captain['Player']} (Score: {vice_captain['Predicted_Score']:.2f})")

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.error("Full traceback:")
        st.code(traceback.format_exc())