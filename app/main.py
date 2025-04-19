import streamlit as st
import pandas as pd
from utils import load_pokemon_data, get_pokemon_stats, predict_winner

st.set_page_config(page_title="Pokémon Battle Predictor", layout="wide")
st.title("⚔️ Competitive Pokémon Battle Predictor")

# Load full Pokémon data
pokemon_df = load_pokemon_data()

# Get unique Pokémon types
types = sorted(set(pokemon_df["Type 1"].unique()).union(set(pokemon_df["Type 2"].dropna().unique())))

# Type filter sidebar
st.sidebar.header("🔍 Filter by Pokémon Type")
selected_type_1 = st.sidebar.selectbox("Filter Pokémon 1 by type", ["All"] + types)
selected_type_2 = st.sidebar.selectbox("Filter Pokémon 2 by type", ["All"] + types)

# Show Clear History button in sidebar only after at least one prediction
if "match_history" not in st.session_state:
    st.session_state.match_history = []

if st.session_state.match_history:
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.match_history.clear()
        st.sidebar.success("Match history cleared!")

# Function to filter by selected type
def filter_by_type(df, selected_type):
    if selected_type == "All":
        return df
    return df[(df["Type 1"] == selected_type) | (df["Type 2"] == selected_type)]

filtered_df_1 = filter_by_type(pokemon_df, selected_type_1)
filtered_df_2 = filter_by_type(pokemon_df, selected_type_2)

# Columns for Pokémon selection and stats
col1, col2 = st.columns(2)

with col1:
    pokemon_1 = st.selectbox("Select Pokémon 1", filtered_df_1["Name"].tolist())
    stats_1 = get_pokemon_stats(pokemon_df, pokemon_1)
    st.write("### Stats for Pokémon 1")
    st.dataframe(stats_1)

with col2:
    pokemon_2 = st.selectbox("Select Pokémon 2", filtered_df_2["Name"].tolist(), index=0)
    stats_2 = get_pokemon_stats(pokemon_df, pokemon_2)
    st.write("### Stats for Pokémon 2")
    st.dataframe(stats_2)

# Prediction logic
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

        # Log match to session history
        st.session_state.match_history.append({
            "Pokémon 1": pokemon_1,
            "Pokémon 2": pokemon_2,
            "Predicted Winner": winner,
            f"{pokemon_1} Win %": round(proba[1] * 100, 2),
            f"{pokemon_2} Win %": round(proba[0] * 100, 2)
        })

# Display match history with index starting at 1
if st.session_state.match_history:
    st.write("## 📋 Match History")
    history_df = pd.DataFrame(st.session_state.match_history)
    history_df.index = range(1, len(history_df) + 1)
    history_df.index.name = "Match #"
    st.dataframe(history_df)
