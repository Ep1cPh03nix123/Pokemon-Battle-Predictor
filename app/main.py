import streamlit as st
import pandas as pd
from utils import load_pokemon_data, get_pokemon_stats, predict_winner

st.set_page_config(page_title="Pokémon Battle Predictor", layout="wide")
st.title("⚔️ Competitive Pokémon Battle Predictor")

# Load data
pokemon_df = load_pokemon_data()

# Dropdowns to select Pokémon
col1, col2 = st.columns(2)

with col1:
    pokemon_1 = st.selectbox("Select Pokémon 1", pokemon_df["Name"].tolist())
    stats_1 = get_pokemon_stats(pokemon_df, pokemon_1)
    st.write("### Stats for Pokémon 1")
    st.dataframe(stats_1)

with col2:
    pokemon_2 = st.selectbox("Select Pokémon 2", pokemon_df["Name"].tolist(), index=1)
    stats_2 = get_pokemon_stats(pokemon_df, pokemon_2)
    st.write("### Stats for Pokémon 2")
    st.dataframe(stats_2)

# Prediction button
if st.button("⚔️ Predict Battle Outcome"):
    if pokemon_1 == pokemon_2:
        st.warning("Please choose two different Pokémon!")
    else:
        prediction, proba = predict_winner(pokemon_df, pokemon_1, pokemon_2)
        winner = pokemon_1 if prediction == 1 else pokemon_2

        st.success(f"🏆 Predicted Winner: **{winner}**")
        st.write("### Win Probabilities:")
        st.markdown(f"- **{pokemon_1}:** {round(proba[1] * 100, 2)}%")
        st.markdown(f"- **{pokemon_2}:** {round(proba[0] * 100, 2)}%")