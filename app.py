import streamlit as st
import pandas as pd
import joblib
import base64

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}

h1,h2,h3 {
    color: #00ffd5;
    text-align:center;
}

.stButton>button {
    background-color: #00ffd5;
    color: black;
    border-radius: 12px;
    height: 55px;
    width: 100%;
    font-size: 22px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #00c9a7;
}

.block-container {
    padding-top: 2rem;
}

[data-testid="stMetricValue"] {
    color: #00ffd5;
    font-size: 35px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD MODEL & SCALER
# -----------------------------------

try:

    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")

except:

    st.error("❌ fraud_model.pkl or scaler.pkl not found")
    st.stop()

# -----------------------------------
# TITLE
# -----------------------------------

st.title("💳 Credit Card Fraud Detection System")

st.markdown("""
### Machine Learning Project by Abhay Solanki

This AI system predicts whether a credit card transaction is fraudulent or normal using Machine Learning.
""")

# -----------------------------------
# MAIN INPUTS
# -----------------------------------

st.subheader("📌 Transaction Information")

col1, col2 = st.columns(2)

with col1:

    time = st.number_input(
        "Transaction Time",
        value=10000.0
    )

with col2:

    amount = st.number_input(
        "Transaction Amount",
        value=100.0
    )

# -----------------------------------
# PCA FEATURES
# -----------------------------------

st.subheader("📊 PCA Transaction Features")

feature_values = {}

cols = st.columns(4)

for i in range(1, 29):

    with cols[(i - 1) % 4]:

        feature_values[f'V{i}'] = st.number_input(
            f'V{i}',
            value=0.0,
            key=f'v{i}'
        )

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------

if st.button("🚀 Detect Fraud"):

    # -----------------------------
    # CREATE INPUT DATA
    # -----------------------------

    input_data = {
        'Time': time
    }

    input_data.update(feature_values)

    input_data['Amount'] = amount

    data = pd.DataFrame([input_data])

    # -----------------------------
    # SCALE FEATURES
    # -----------------------------

    data[['Time', 'Amount']] = scaler.transform(
        data[['Time', 'Amount']]
    )

    # -----------------------------
    # PREDICTION
    # -----------------------------

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1] * 100

    # -----------------------------------
    # EXTRA STRONG FRAUD CHECK
    # -----------------------------------

    suspicious = False

    if probability >= 30:
        suspicious = True

    # -----------------------------------
    # RESULT SECTION
    # -----------------------------------

    st.subheader("📢 Prediction Result")

    # -----------------------------------
    # FRAUD / SUSPICIOUS
    # -----------------------------------

    if prediction == 1 or suspicious:

        # FLASHING ALERT
        st.markdown("""
        <h1 style='
        color:red;
        text-align:center;
        animation: blinker 1s linear infinite;'>
        🚨 WARNING! SUSPICIOUS TRANSACTION DETECTED 🚨
        </h1>

        <style>
        @keyframes blinker {
          50% { opacity: 0; }
        }
        </style>
        """, unsafe_allow_html=True)

        st.error("🚨 Fraudulent / Suspicious Transaction Detected!")

        # -----------------------------------
        # AUTO SIREN SOUND
        # -----------------------------------

        audio_file = open("siren.mp3.wav", "rb")
        audio_bytes = audio_file.read()

        audio_base64 = base64.b64encode(audio_bytes).decode()

        audio_html = f"""
        <audio autoplay>
        <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        </audio>
        """

        st.markdown(audio_html, unsafe_allow_html=True)

        # -----------------------------------
        # ANIMATION
        # -----------------------------------

        st.balloons()

    # -----------------------------------
    # NORMAL TRANSACTION
    # -----------------------------------

    else:

        st.success("✅ Normal Transaction")

    # -----------------------------------
    # PROBABILITY
    # -----------------------------------

    st.metric(
        "Fraud Probability",
        f"{probability:.2f}%"
    )

    # Progress bar
    st.progress(min(int(probability), 100))

    # -----------------------------------
    # SHOW DATA
    # -----------------------------------

    with st.expander("📄 View Transaction Data"):

        st.dataframe(data)

# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("---")

st.markdown("""
<center>

💻 Built with Streamlit & Machine Learning ❤️

</center>
""", unsafe_allow_html=True)