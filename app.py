import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Browsing Behavior Analyzer", layout="wide")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    merged = pd.read_csv("merged_browser_ram.csv")
    session = pd.read_csv("session_features.csv")

    merged['timestamp'] = pd.to_datetime(merged['timestamp'])
    session['start_time'] = pd.to_datetime(session['start_time'])
    session['end_time'] = pd.to_datetime(session['end_time'])

    return merged, session

merged_df, session_df = load_data()

# ---------------------------
# LOAD MODELS
# ---------------------------
@st.cache_resource
def load_models():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")

    lstm_model = load_model("lstm_model.h5",compile=False)
    le = joblib.load("label_encoder.pkl")

    autoencoder = load_model("autoencoder_model.h5",compile=False)
    ae_scaler = joblib.load("autoencoder_scaler.pkl")

    return kmeans, scaler, lstm_model, le, autoencoder, ae_scaler

kmeans, scaler, lstm_model, le, autoencoder, ae_scaler = load_models()

# ---------------------------
# TITLE
# ---------------------------
st.title("🧠 Time-Based Browsing Pattern Analyzer")
st.markdown("Analyze user browsing behavior, system RAM usage, and generate intelligent insights.")

# ---------------------------
# SIDEBAR FILTER
# ---------------------------
st.sidebar.header("Filter")

days = st.sidebar.selectbox("Select Time Window", ["All", "3 Days", "5 Days"])

if days != "All":
    n = int(days.split()[0])
    cutoff = merged_df['timestamp'].max() - pd.Timedelta(days=n)
    merged_df = merged_df[merged_df['timestamp'] >= cutoff]

# ---------------------------
# OVERVIEW
# ---------------------------
st.header("📊 Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Events", len(merged_df))
col2.metric("Total Sessions", session_df['session_id'].nunique())
col3.metric("Avg Session Duration (min)", round(session_df['session_duration'].mean(), 2))

# ---------------------------
# CATEGORY ANALYSIS
# ---------------------------
st.header("🌐 Category Distribution")

cat_counts = merged_df['category'].value_counts()

fig, ax = plt.subplots()
cat_counts.plot(kind='bar', ax=ax)
st.pyplot(fig)

st.markdown("""
**Insight:**  
This chart shows dominant user activity categories such as social, video, or learning.  
It helps identify where most time is spent.
""")

# ---------------------------
# TIME ANALYSIS
# ---------------------------
st.header("⏰ Time-Based Usage")

merged_df['hour'] = merged_df['timestamp'].dt.hour
hour_counts = merged_df.groupby('hour').size()

fig, ax = plt.subplots()
hour_counts.plot(kind='bar', ax=ax)
st.pyplot(fig)

st.markdown("""
**Insight:**  
Peak browsing hours highlight productivity or distraction patterns.
""")

# ---------------------------
# RAM ANALYSIS
# ---------------------------
st.header("💻 RAM Usage by Category")

ram_usage = merged_df.groupby('category')['browser_ram_mb'].mean().sort_values()

fig, ax = plt.subplots()
ram_usage.plot(kind='barh', ax=ax)
st.pyplot(fig)

st.markdown("""
**Insight:**  
Categories with higher RAM usage (e.g., video) impact system performance significantly.
""")

# ---------------------------
# CLUSTER ANALYSIS
# ---------------------------
st.header("🔍 Session Clustering")

cluster_summary = session_df.groupby('cluster')[[
    'session_duration',
    'num_events',
    'avg_browser_ram'
]].mean()

st.dataframe(cluster_summary)

st.markdown("""
**Explanation:**  
Clustering groups sessions into behavior types:
- Short sessions → casual browsing  
- Long sessions → binge or focused usage  
- High RAM sessions → heavy applications  

This helps identify user behavior patterns.
""")

# ---------------------------
# ANOMALY DETECTION
# ---------------------------
st.header("🚨 Anomaly Detection")

if 'anomaly' in session_df.columns:
    anomalies = session_df[session_df['anomaly'] == True]
    st.write("Total anomalies:", len(anomalies))
    st.dataframe(anomalies.head(10))

    st.markdown("""
    **Explanation:**  
    Anomalies represent unusual sessions such as:
    - Extremely long sessions  
    - High RAM spikes  
    - Irregular browsing behavior  
    """)
else:
    st.warning("Run autoencoder to detect anomalies.")

# ---------------------------
# PREDICTION SECTION
# ---------------------------
st.header("🔮 Predict User Behavior")

col1, col2 = st.columns(2)

with col1:
    duration = st.number_input("Session Duration (min)", value=30)
    events = st.number_input("Number of Events", value=10)
    avg_ram = st.number_input("Avg RAM", value=800)
    peak_ram = st.number_input("Peak RAM", value=1200)


    seq_input = st.text_input("Last 5 categories (comma separated)",
                             "social,video,learning,social,video")

analyze = st.button("Analyze")

if analyze:

    session_input = [duration, events, avg_ram, peak_ram]

    # ---- Cluster ----
    scaled = scaler.transform([session_input])
    cluster = kmeans.predict(scaled)[0]

    # ---- LSTM ----
    seq = [s.strip() for s in seq_input.split(",")]

    try:
        encoded = [le.transform([c])[0] for c in seq]
    except:
        st.error("Invalid category entered")
        st.stop()

    pred = lstm_model.predict(np.array([encoded]))
    next_cat = le.inverse_transform([np.argmax(pred)])[0]

    # ---- Autoencoder ----
    scaled_ae = ae_scaler.transform([session_input])
    recon = autoencoder.predict(scaled_ae)

    error = np.mean((scaled_ae - recon) ** 2)

    threshold = 0.05
    anomaly = error > threshold

    # ---- Recommendations ----
    recs = []

    if peak_ram > 1200:
        recs.append("⚠️ High RAM usage → close unused tabs")

    if duration > 60:
        recs.append("⏳ Long session → take breaks")

    if events > 30:
        recs.append("🔁 Too many events → reduce multitasking")

    if next_cat == "social":
        recs.append("📱 Social usage predicted → avoid distractions")

    if anomaly:
        recs.append("🚨 Unusual session detected")

    if not recs:
        recs.append("✅ Usage looks healthy")

    # ---- Output ----
    st.subheader("Results")

    st.write("Cluster:", cluster)
    st.write("Next Category:", next_cat)
    st.write("Anomaly:", anomaly)

    st.subheader("📌 Recommendations")

    for r in recs:
        st.success(r)

    # DEBUG
    st.write("DEBUG:", error)
