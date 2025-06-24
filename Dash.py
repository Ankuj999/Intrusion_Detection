import streamlit as st
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
import base64

def set_background(local_image_file):
    with open(local_image_file, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("Anomaly-Detection.jpg")


# Load model
model = joblib.load("D:\\SI Rourkela\\Jup\\xgboost_binary_model.pkl")

st.markdown("""
<h1 style='color: #00FFFF;'>🚀 Welcome to the Real-Time IoMT Anomaly Detection Dashboard</h1>
<h3 style='color: #FF9800;'>🔍 Upload your preprocessed dataset to begin real-time anomaly monitoring</h3>
""", unsafe_allow_html=True)

st.markdown("""
<div style='
    background-color: rgba(255, 0, 0, 0.1);
    border-left: 5px solid #FF0000;
    padding: 15px;
    border-radius: 8px;
    color: #B22222	;
    font-weight: 600;
'>
🚨 <b>Warning:</b> Please make sure your dataset is preprocessed correctly before uploading.<br><br>
- All features must match the structure used during model training.<br>
- For reference check the <b>UNSW_NB15</b> Dataset.<br>
- The file must contain a <b>'label'</b> column for ground truth comparison.<br>
- Missing values should be handled.<br>
- No extra or missing columns.<br>
- The model expects numerical features only (already encoded).
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p style='color:#FFE4E1 ; font-size:24px; font-weight:bold; text-align:center;'>
📂 Upload your dataset (CSV with a 'label' column)
</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["csv"])


# if uploaded_file is None:
#     st.title("🚀 Welcome to the Real-Time IoMT Anomaly Detection Dashboard")
#     st.subheader("🔍 Upload your preprocessed dataset to begin real-time anomaly monitoring")
#     st.markdown("---")

# st.markdown("""
# <div style='
#     background-color: rgba(255, 0, 0, 0.1);
#     border-left: 5px solid #FF0000;
#     padding: 15px;
#     border-radius: 8px;
#     color: #B22222	;
#     font-weight: 600;
# '>
# 🚨 <b>Warning:</b> Please make sure your dataset is preprocessed correctly before uploading.<br><br>
# - All features must match the structure used during model training.<br>
# - For reference check the <b>UNSW_NB15</b> Dataset.<br>
# - The file must contain a <b>'label'</b> column for ground truth comparison.<br>
# - Missing values should be handled.<br>
# - No extra or missing columns.<br>
# - The model expects numerical features only (already encoded).
# </div>
# """, unsafe_allow_html=True)


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Validate label column
    if "label" not in df.columns:
        st.error("❌ The uploaded CSV must contain a 'label' column.")
        st.stop()

    X = df.drop("label", axis=1)
    y_true = df["label"].values
else:
    st.markdown("""
    <div style='
        background-color: #f9f9ff;
        color: #4444aa;
        border-left: 6px solid #8888ff;
        padding: 12px;
        border-radius: 6px;
        font-weight: bold;
    '>
    📌 Please upload a CSV file to begin.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Session state setup
if "i" not in st.session_state:
    st.session_state.i = 0
    st.session_state.anomaly_count = 0
    st.session_state.anomaly_indexes = []
    st.session_state.confidences = []

st.markdown("<h1 style='color:#3CB371;'>🔍 Anomaly Detection Dashboard</h1>", unsafe_allow_html=True)
# st.markdown("Streaming predictions for Anomaly or Intrusion from uploaded Dataset of IoMT network...")

placeholder = st.empty()

# Real-time prediction loop
if st.session_state.i < len(X):
    row = X.iloc[st.session_state.i].values.reshape(1, -1)
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0][1]
    actual = y_true[st.session_state.i]

    # Store confidence for every row
    st.session_state.confidences.append(prob)

    # Store anomaly info
    if pred == 1:
        st.session_state.anomaly_count += 1
        st.session_state.anomaly_indexes.append(st.session_state.i)

    # UI display
    with placeholder.container():
        st.markdown(
        f"<h3 style='color: #FFDDE2;'>▶️ Message {st.session_state.i + 1}/{len(X)}</h3>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<p style='color: #00ffff; font-size: 18px; font-weight: 600;'>"
        f"🔍 <b>Prediction:</b> {'Anomaly' if pred == 1 else 'Normal'}</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<p style='color: #FFD700; font-size: 18px; font-weight: 600;'>"
        f"✅ <b>Actual:</b> {'Anomaly' if actual == 1 else 'Normal'}</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<p style='color: #00FF99; font-size: 18px; font-weight: 600;'>"
        f"📊 <b>Confidence Score:</b> {prob:.3f}</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style='
            background-color: #ffe6e6;
            color: #b30000;
            border-left: 6px solid #cc0000;
            padding: 12px;
            margin-top: 10px;
            border-radius: 6px;
            font-weight: bold;
        '>
        🚨 Total Anomalies Detected: {st.session_state.anomaly_count}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Plot
    fig, ax = plt.subplots()
    ax.plot(st.session_state.confidences, color="orange", label="Anomaly Confidence")
    ax.axhline(0.5, color='red', linestyle='--', label='Threshold')
    ax.set_xlabel("Message Index")
    ax.set_ylabel("Confidence")
    ax.set_title("📈 Model Confidence Over Time")
    ax.legend()
    st.pyplot(fig)


    # Step forward
    st.session_state.i += 1
    time.sleep(0.01)
    st.rerun()

# Final summary
else:
    st.markdown(
        "<div style='background-color:#e6fff5; border-left: 5px solid #3CB371; padding: 15px; border-radius: 8px;'>"
        "<h4 style='color:#2e8b57;'>✅ All messages processed.</h4>"
        "</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h3 style='color:#3CB371;'>🧾 Summary Report</h3>", 
        unsafe_allow_html=True
    )

    st.markdown(
        f"<p style='color:#3CB371;'>- 🔢 <b>Total Anomalies Detected:</b> <span style='color:#FF4500;'>{st.session_state.anomaly_count}</span></p>", 
        unsafe_allow_html=True
    )

    st.markdown(
        f"<p style='color:#3CB371;'>- 📍 <b>Indexes of Anomalies Detected:</b></p>", 
        unsafe_allow_html=True
    )

    st.code(st.session_state.anomaly_indexes, language="python")
     # Pie chart of prediction results
    st.markdown("<h3 style='color:#FFFFF0;'>🥧 Anomaly vs Normal Predictions",unsafe_allow_html=True)
    labels = ["Normal", "Anomaly"]
    sizes = [len(X) - st.session_state.anomaly_count, st.session_state.anomaly_count]
    colors = ["#90ee90", "#ff6347"]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)
st.markdown("""
<div style='background-color:#f9f9f9; padding:10px; border-top:1px solid #ccc; text-align:center; font-size:14px; color:#444;'>
    🚀 Designed & Developed by <b>Ankuj Saha</b><br>
    <i>Data Analyst | Machine Learning Developer</i>
</div>
""", unsafe_allow_html=True)