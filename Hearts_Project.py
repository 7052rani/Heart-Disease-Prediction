import base64
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

# Note: You'll need to load your model and scaler
# model = joblib.load('model.pkl')  # or your model loading method
# scaler = joblib.load('scaler.pkl')  # or your scaler loading method

def get_base64(file):
    with open(file,"rb") as f:
        return base64.b64encode(f.read()).decode()

bg=get_base64("Screenshot 2026-08-02 221249.png")

page_bg=f"""
<style>

.stApp{{
background-image:url("data:image/jpg;base64,{bg}");
background-size:cover;
background-position:center;
background-attachment:fixed;
}}

[data-testid="stHeader"]{{
background:rgba(0,0,0,0);
}}

[data-testid="stSidebar"]{{
background:rgba(20,20,20,.75);
backdrop-filter:blur(15px);
}}

.main-card{{
background:rgba(255,255,255,.15);
backdrop-filter:blur(18px);
padding:25px;
border-radius:25px;
border:1px solid rgba(255,255,255,.2);
box-shadow:0px 0px 25px rgba(0,0,0,.35);
}}

.big-title{{
text-align:center;
font-size:52px;
color:white;
font-weight:bold;
}}

.subtitle{{
text-align:center;
color:#dddddd;
font-size:20px;
}}

.stButton>button{{
width:100%;
height:60px;
border-radius:18px;
font-size:22px;
font-weight:bold;
border:none;
background:linear-gradient(90deg,#ff416c,#ff4b2b);
color:white;
}}

.stButton>button:hover{{
transform:scale(1.04);
transition:.3s;
}}

</style>
"""

st.markdown(page_bg,unsafe_allow_html=True)

# ==================== SIDEBAR ====================
st.sidebar.image("Screenshot 2026-08-02 221434.png", width=180)
st.sidebar.title("🏥 AI Healthcare")
st.sidebar.markdown("---")
st.sidebar.success("✔ KNN Machine Learning Model")
st.sidebar.info("""
Fill all patient details carefully.

Click **Predict** to know the risk.
""")
st.sidebar.markdown("---")
st.sidebar.write("👩‍💻 Developed by")
st.sidebar.write("### Rani Rajput")

# ==================== MAIN CONTENT ====================
col1,col2=st.columns([1,6])

with col1:
    st.image("Screenshot 2026-08-02 221400.png",width=200)

with col2:
    st.markdown("""
<div class="big-title">

❤️ Heart Disease Prediction

</div>

<div class="subtitle">

Artificial Intelligence Based Heart Healthcare System

</div>

""",unsafe_allow_html=True)
    st.markdown('<div class="main-card">',unsafe_allow_html=True)

# Main content
st.markdown("""
<div class="main-card">
<h2 style='color:white;'>👨 Patient Information</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider(
        "🎂 Age",
        18,
        100,
        40
    )

    sex = st.selectbox(
        "🚻 Gender",
        ["M", "F"]
    )

    fasting_bs = st.selectbox(
        "🩸 Fasting Blood Sugar",
        [0, 1]
    )

with col2:
    chest_pain = st.selectbox(
        "❤️ Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_bp = st.number_input(
        "🩺 Resting Blood Pressure",
        80,
        200,
        120
    )

    cholesterol = st.number_input(
        "🧪 Cholesterol",
        100,
        600,
        200
    )

with col3:
    resting_ecg = st.selectbox(
        "📈 Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "💓 Max Heart Rate",
        60,
        220,
        150
    )

    exercise_angina = st.selectbox(
        "🏃 Exercise Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "📉 Old Peak",
        0.0,
        6.0,
        1.0
    )

    st_slope = st.selectbox(
        "📊 ST Slope",
        ["Up", "Flat", "Down"]
    )

# Predict button
predict = st.button(
    "🩺 Predict Heart Disease",
    use_container_width=True
)

# If you have model and scaler loaded, uncomment this section
if predict:
    with st.spinner("🧠 AI is analyzing patient data..."):
        time.sleep(2)
    
    # Define expected columns (this should match your model's training columns)
    # You need to define this based on your model
    expected_columns = []  # Define your expected columns here
    
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    # Comment this out until you have your model loaded
    # scaled_input = scaler.transform(input_df)

    # with st.spinner("🧠 AI is analysing patient..."):
    #     prediction = model.predict(scaled_input)[0]
    #     probability = model.predict_proba(scaled_input)
    #     risk = probability[0][1]
    
    # For demo purposes
    prediction = 0  # Remove this when you have actual model
    risk = 0.25     # Remove this when you have actual model
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk*100,
        title={
            'text':"Heart Disease Risk (%)",
            'font':{'size':30}
        },
        gauge={
            'axis':{'range':[0,100]},
            'bar':{'color':"red"},
            'steps':[
                {'range':[0,30],'color':'green'},
                {'range':[30,70],'color':'orange'},
                {'range':[70,100],'color':'red'}
            ]
        }
    ))

    st.plotly_chart(fig,use_container_width=True)
    
    st.metric(
        "Risk Probability",
        f"{risk*100:.2f}%"
    )
    
    # FIXED: Proper indentation for error/success messages
    if prediction == 1:
        st.error("""
🚨 High Risk of Heart Disease

Please consult a Cardiologist immediately.
""")
    else:
        st.success("""
✅ Low Risk of Heart Disease

Heart condition looks normal.
""")
    
    st.markdown("## 📋 Patient Summary")
    
    summary = pd.DataFrame({
        "Feature":[
            "Age",
            "Gender",
            "Chest Pain",
            "Blood Pressure",
            "Cholesterol",
            "Max Heart Rate"
        ],
        "Value":[
            age,
            sex,
            chest_pain,
            resting_bp,
            cholesterol,
            max_hr
        ]
    })

    st.dataframe(summary,use_container_width=True)
    st.progress(risk)
    
    if prediction == 0:
        st.balloons()

st.markdown("---")
st.caption(" AI Heart Healthcare Dashboard")