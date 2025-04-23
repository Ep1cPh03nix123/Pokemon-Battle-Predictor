import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import load_pokemon_data, get_pokemon_stats, predict_winner

st.set_page_config(page_title="Pokémon Battle Predictor", layout="wide")
st.title("⚔️ Competitive Pokémon Battle Predictor")

# Load Pokémon data
pokemon_df = load_pokemon_data()

# Unique types
types = sorted(set(pokemon_df["Type 1"].unique()).union(set(pokemon_df["Type 2"].dropna().unique())))

# Sidebar type filter
st.sidebar.header("🔍 Filter by Pokémon Type")
selected_type_1 = st.sidebar.selectbox("Filter Pokémon 1 by type", ["All"] + types)
selected_type_2 = st.sidebar.selectbox("Filter Pokémon 2 by type", ["All"] + types)

# Match history session init
if "match_history" not in st.session_state:
    st.session_state.match_history = []

# Clear history option
if st.session_state.match_history:
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.match_history.clear()
        st.sidebar.success("Match history cleared!")

# Type filtering logic
def filter_by_type(df, selected_type):
    if selected_type == "All":
        return df
    return df[(df["Type 1"] == selected_type) | (df["Type 2"] == selected_type)]

filtered_df_1 = filter_by_type(pokemon_df, selected_type_1)
filtered_df_2 = filter_by_type(pokemon_df, selected_type_2)

# Stat color scheme
stat_names = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
stat_colors = {
    "HP": "#FF4C4C", "Attack": "#FF9900", "Defense": "#FFDD57",
    "Sp. Atk": "#4C9AFF", "Sp. Def": "#B266FF", "Speed": "#2ECC71"
}

# Pokémon selectors and stat display
col1, col2 = st.columns(2)
with col1:
    pokemon_1 = st.selectbox("Select Pokémon 1", filtered_df_1["Name"].tolist())
    stats_1 = get_pokemon_stats(pokemon_df, pokemon_1)
    st.write("### Stats for Pokémon 1")
    stat_cols = st.columns(6)
    for i, stat in enumerate(stat_names):
        value = int(stats_1[stat].values[0])
        color = stat_colors[stat]
        stat_cols[i].markdown(
            f"<div style='text-align:center'><b style='color:{color}'>{stat}</b><br><span style='font-size:24px'>{value}</span></div>",
            unsafe_allow_html=True
        )

with col2:
    pokemon_2 = st.selectbox("Select Pokémon 2", filtered_df_2["Name"].tolist(), index=0)
    stats_2 = get_pokemon_stats(pokemon_df, pokemon_2)
    st.write("### Stats for Pokémon 2")
    stat_cols = st.columns(6)
    for i, stat in enumerate(stat_names):
        value = int(stats_2[stat].values[0])
        color = stat_colors[stat]
        stat_cols[i].markdown(
            f"<div style='text-align:center'><b style='color:{color}'>{stat}</b><br><span style='font-size:24px'>{value}</span></div>",
            unsafe_allow_html=True
        )

# Radar chart setup
stats_1_values = [int(stats_1[stat].values[0]) for stat in stat_names]
stats_2_values = [int(stats_2[stat].values[0]) for stat in stat_names]
stats_1_values += [stats_1_values[0]]
stats_2_values += [stats_2_values[0]]
categories = stat_names + [stat_names[0]]

fig = go.Figure()
fig.add_trace(go.Scatterpolar(r=stats_1_values, theta=categories, fill='toself', name=pokemon_1, line=dict(color='#FF4C4C'), fillcolor='rgba(255, 76, 76, 0.3)'))
fig.add_trace(go.Scatterpolar(r=stats_2_values, theta=categories, fill='toself', name=pokemon_2, line=dict(color='#4C9AFF'), fillcolor='rgba(76, 154, 255, 0.3)'))
fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, max(stats_1_values + stats_2_values) + 10])),
    showlegend=True,
    title="📊 Stat Comparison"
)
st.plotly_chart(fig)

# Prediction logic
if st.button("⚔️ Predict Battle Outcome"):
    if pokemon_1 == pokemon_2:
        st.warning("Please choose two different Pokémon!")
    else:
        prediction, proba = predict_winner(pokemon_df, pokemon_1, pokemon_2)
        winner = pokemon_1 if prediction == 1 else pokemon_2
        win_diff = abs(round(proba[1] * 100, 2) - round(proba[0] * 100, 2))

        st.success(f"🏆 Predicted Winner: **{winner}**")
        st.write("### Win Probabilities:")
        st.markdown(f"- **{pokemon_1}:** {round(proba[1] * 100, 2)}%")
        st.markdown(f"- **{pokemon_2}:** {round(proba[0] * 100, 2)}%")
        st.caption(f"💡 *Model used: XGBoost with Type Advantage (TypeEff_Diff). Win gap: {win_diff}%*")

        # Add to match history
        st.session_state.match_history.append({
            "Pokémon 1": pokemon_1,
            "Pokémon 2": pokemon_2,
            "Predicted Winner": winner,
            f"{pokemon_1} Win %": round(proba[1] * 100, 2),
            f"{pokemon_2} Win %": round(proba[0] * 100, 2)
        })

# Display match history
if st.session_state.match_history:
    st.write("## 📋 Match History")
    history_df = pd.DataFrame(st.session_state.match_history)
    history_df.index = range(1, len(history_df) + 1)
    history_df.index.name = "Match #"
    st.dataframe(history_df)
