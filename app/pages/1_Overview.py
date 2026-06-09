import plotly.express as px
import streamlit as st

from src.dashboard_data import load_dashboard_data

st.set_page_config(page_title="Overview - Disaster Response DSS", layout="wide")
st.title("Disaster Response Overview")
st.caption(
    "Operational simulation on all 2,237 rows of the official test split. "
    "Formal model metrics use the leakage-safe subset shown on Model Evaluation."
)

df = load_dashboard_data()

total = len(df)
informative_rate = (df["true_label"] == "informative").mean() * 100
high_priority = (df["priority"] == "High").sum()
manual_review = df["manual_review"].astype(bool).sum()

columns = st.columns(4)
columns[0].metric("Test posts", f"{total:,}")
columns[1].metric("Informative ground truth", f"{informative_rate:.1f}%")
columns[2].metric("High priority", f"{high_priority:,}")
columns[3].metric("Manual review", f"{manual_review:,}")

left, right = st.columns(2)
with left:
    priority = df["priority"].value_counts().rename_axis("Priority").reset_index(name="Count")
    figure = px.pie(
        priority,
        names="Priority",
        values="Count",
        color="Priority",
        color_discrete_map={"Low": "#2E7D32", "Medium": "#ED6C02", "High": "#C62828"},
        hole=0.45,
        title="Priority distribution",
    )
    st.plotly_chart(figure, width="stretch")

with right:
    teams = df["assigned_team"].value_counts().rename_axis("Team").reset_index(name="Count")
    figure = px.bar(
        teams,
        x="Count",
        y="Team",
        orientation="h",
        title="Routing workload by response team",
        color="Team",
    )
    figure.update_layout(showlegend=False)
    st.plotly_chart(figure, width="stretch")

left, right = st.columns(2)
with left:
    categories = (
        df["fused_category"].value_counts().rename_axis("Category").reset_index(name="Count")
    )
    figure = px.bar(
        categories,
        x="Count",
        y="Category",
        orientation="h",
        title="Predicted humanitarian categories",
        color="Count",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(figure, width="stretch")

with right:
    figure = px.histogram(
        df,
        x="risk_score",
        color="priority",
        nbins=25,
        title="Risk-score distribution",
        color_discrete_map={"Low": "#2E7D32", "Medium": "#ED6C02", "High": "#C62828"},
    )
    st.plotly_chart(figure, width="stretch")
