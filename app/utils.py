import pandas as pd

def load_pokemon_data():
    df = pd.read_csv("data/pokemon.csv")
    df.rename(columns={"#": "ID"}, inplace=True)
    return df

def get_pokemon_stats(df, name):
    row = df[df["Name"] == name]
    return row[["Type 1", "Type 2", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]
