import streamlit as st

from src.dashboard_data import load_dashboard_data

st.set_page_config(page_title="Priority Queue - Disaster Response DSS", layout="wide")
st.title("Disaster Response Priority Queue")
st.caption("Held-out posts sorted by the prescriptive DSS risk score.")

queue = load_dashboard_data()

with st.sidebar:
    st.header("Queue filters")
    priority = st.selectbox("Priority", ["All", "High", "Medium", "Low"])
    team = st.selectbox(
        "Assigned team", ["All"] + sorted(queue["assigned_team"].unique())
    )
    event = st.selectbox(
        "Disaster event", ["All"] + sorted(queue["event_name"].unique())
    )
    review_only = st.checkbox("Manual review only")

filtered = queue.copy()
if priority != "All":
    filtered = filtered[filtered["priority"] == priority]
if team != "All":
    filtered = filtered[filtered["assigned_team"] == team]
if event != "All":
    filtered = filtered[filtered["event_name"] == event]
if review_only:
    filtered = filtered[filtered["manual_review"].astype(bool)]

columns = st.columns(4)
columns[0].metric("Visible posts", len(filtered))
columns[1].metric("High priority", int((filtered["priority"] == "High").sum()))
columns[2].metric(
    "Emergency team",
    int(filtered["assigned_team"].str.contains("Emergency Team").sum()),
)
columns[3].metric(
    "Manual review", int(filtered["manual_review"].astype(bool).sum())
)

visible_columns = [
    "tweet_id",
    "event_name",
    "tweet_text",
    "risk_score",
    "priority",
    "assigned_team",
    "manual_review",
]
st.dataframe(
    filtered[visible_columns],
    hide_index=True,
    width="stretch",
    height=470,
)
st.download_button(
    "Download filtered queue",
    filtered.to_csv(index=False).encode("utf-8"),
    "disaster_priority_queue.csv",
    "text/csv",
)

if not filtered.empty:
    selected_id = st.selectbox("Inspect Tweet ID", filtered["tweet_id"].tolist())
    item = filtered[filtered["tweet_id"] == selected_id].iloc[0]
    left, right = st.columns([1, 2])
    with left:
        st.metric("Risk score", item["risk_score"])
        st.write(f"**Priority:** {item['priority']}")
        st.write(f"**Team:** {item['assigned_team']}")
        st.write(f"**Predicted category:** {item['fused_category']}")
        st.write(f"**Conflict score:** {item['conflict_score']:.3f}")
    with right:
        st.write("**Original post**")
        st.write(item["tweet_text"])
        st.write("**Recommended action**")
        st.success(item["recommended_action"])
        if bool(item["manual_review"]):
            st.warning(item["review_reason"])
