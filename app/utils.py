import pandas as pd
import joblib
import xgboost as xgb

# --- Load assets ---
def load_pokemon_data():
    df = pd.read_csv("data/pokemon.csv")
    df.rename(columns={"#": "ID"}, inplace=True)
    return df

def get_pokemon_stats(df, name):
    row = df[df["Name"] == name]
    return row[["Type 1", "Type 2", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
type_chart = pd.read_csv("data/Pokemon Type Chart.csv")

# --- Base stat calculation (Level 50, IV=31, EV=0, Neutral Nature) ---
def calculate_stat(base, level=50, iv=31, ev=0, nature=1.0, is_hp=False):
    if is_hp:
        return int((((2 * base + iv + (ev // 4)) * level) / 100) + level + 10)
    else:
        return int(((((2 * base + iv + (ev // 4)) * level) / 100) + 5) * nature)

# --- Type effectiveness ---
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

# --- 1v1 prediction ---
def predict_winner(pokemon_df, name1, name2):
    p1 = pokemon_df[pokemon_df["Name"] == name1].iloc[0]
    p2 = pokemon_df[pokemon_df["Name"] == name2].iloc[0]

    # Calculate real-world stats without Speed
    stats_1 = [
        calculate_stat(p1["HP"], is_hp=True),
        calculate_stat(p1["Attack"]),
        calculate_stat(p1["Defense"]),
        calculate_stat(p1["Sp. Atk"]),
        calculate_stat(p1["Sp. Def"]),
    ]

    stats_2 = [
        calculate_stat(p2["HP"], is_hp=True),
        calculate_stat(p2["Attack"]),
        calculate_stat(p2["Defense"]),
        calculate_stat(p2["Sp. Atk"]),
        calculate_stat(p2["Sp. Def"]),
    ]

    # BST without Speed
    bst1 = sum(stats_1)
    bst2 = sum(stats_2)

    # Type effectiveness
    p1_types = [p1["Type 1"]] + ([p1["Type 2"]] if pd.notna(p1["Type 2"]) else [])
    p2_types = [p2["Type 1"]] + ([p2["Type 2"]] if pd.notna(p2["Type 2"]) else [])
    p1_eff = compute_type_effectiveness(p1_types, p2_types, type_chart)
    p2_eff = compute_type_effectiveness(p2_types, p1_types, type_chart)
    type_eff_diff = p1_eff - p2_eff

    # Final feature vector (5+5+2+1 = 13 total)
    features = stats_1 + stats_2 + [bst1, bst2, type_eff_diff]
    X = scaler.transform([features])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return prediction, proba

# --- 6v6 team battle ---
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
