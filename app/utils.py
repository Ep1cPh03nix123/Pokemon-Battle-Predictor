import pandas as pd
import joblib

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

def predict_winner(pokemon_df, name1, name2):
    p1 = pokemon_df[pokemon_df["Name"] == name1].iloc[0]
    p2 = pokemon_df[pokemon_df["Name"] == name2].iloc[0]

    # Calculate Base Stat Totals
    bst1 = p1["HP"] + p1["Attack"] + p1["Defense"] + p1["Sp. Atk"] + p1["Sp. Def"] + p1["Speed"]
    bst2 = p2["HP"] + p2["Attack"] + p2["Defense"] + p2["Sp. Atk"] + p2["Sp. Def"] + p2["Speed"]

    # Full feature list (14 base stats + 6 differential features)
    features = [
        # Base stats for Pokémon 1 and 2
        p1["HP"], p1["Attack"], p1["Defense"], p1["Sp. Atk"], p1["Sp. Def"], p1["Speed"],
        p2["HP"], p2["Attack"], p2["Defense"], p2["Sp. Atk"], p2["Sp. Def"], p2["Speed"],
        bst1, bst2,

        # Differential features
        p1["HP"] - p2["HP"],
        p1["Attack"] - p2["Defense"],
        p1["Sp. Atk"] - p2["Sp. Def"],
        p1["Sp. Def"] - p2["Sp. Atk"],
        p1["Speed"] - p2["Speed"],
        bst1 - bst2
    ]

    # Scale and predict
    X = scaler.transform([features])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return prediction, proba
