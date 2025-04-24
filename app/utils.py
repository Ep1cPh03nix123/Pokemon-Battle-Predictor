import pandas as pd
import joblib
import xgboost as xgb

def load_pokemon_data():
    df = pd.read_csv("data/pokemon.csv")
    df.rename(columns={"#": "ID"}, inplace=True)
    return df

def get_pokemon_stats(df, name):
    row = df[df["Name"] == name]
    return row[["Type 1", "Type 2", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]

# Load trained model and scaler
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Load type chart
type_chart = pd.read_csv("data/Pokemon Type Chart.csv")

def compute_type_effectiveness(attacker_types, defender_types, chart_df):
    effectiveness = 1.0
    for atk_type in attacker_types:
        for def_type in defender_types:
            try:
                multiplier = chart_df.loc[chart_df['Attacking'] == atk_type, def_type].values[0]
                effectiveness *= multiplier
            except (KeyError, IndexError):
                effectiveness *= 1.0
    return effectiveness

def predict_winner(pokemon_df, name1, name2):
    p1 = pokemon_df[pokemon_df["Name"] == name1].iloc[0]
    p2 = pokemon_df[pokemon_df["Name"] == name2].iloc[0]

    # Calculate BST
    bst1 = p1[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]].sum()
    bst2 = p2[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]].sum()

    # Types
    p1_types = [p1["Type 1"]] + ([p1["Type 2"]] if pd.notna(p1["Type 2"]) else [])
    p2_types = [p2["Type 1"]] + ([p2["Type 2"]] if pd.notna(p2["Type 2"]) else [])

    # Type effectiveness
    p1_eff = compute_type_effectiveness(p1_types, p2_types, type_chart)
    p2_eff = compute_type_effectiveness(p2_types, p1_types, type_chart)

    # Difference in effectiveness
    type_eff_diff = p1_eff - p2_eff

    # Final features (15 total: 12 base, 2 BST, 1 TypeEff_Diff)
    features = [
        p1["HP"], p1["Attack"], p1["Defense"], p1["Sp. Atk"], p1["Sp. Def"], p1["Speed"],
        p2["HP"], p2["Attack"], p2["Defense"], p2["Sp. Atk"], p2["Sp. Def"], p2["Speed"],
        bst1, bst2,
        type_eff_diff
    ]

    # Scale and predict
    X = scaler.transform([features])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return prediction, proba

# NEW: Team Battle Prediction
def predict_team_battle(pokemon_df, team1, team2):
    team1_score = 0
    team2_score = 0
    total = 0

    for p1 in team1:
        for p2 in team2:
            prediction, _ = predict_winner(pokemon_df, p1, p2)
            if prediction == 1:
                team1_score += 1
            else:
                team2_score += 1
            total += 1

    team1_win_rate = round((team1_score / total) * 100, 2)
    team2_win_rate = round((team2_score / total) * 100, 2)

    return team1_score, team2_score, team1_win_rate, team2_win_rate
