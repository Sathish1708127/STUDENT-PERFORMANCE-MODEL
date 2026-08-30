import streamlit as st
import pandas as pd
import joblib

# Inject Custom HTML and CSS
st.markdown("""
    <style>
    .main-header {
        color: #2E86C1;
        font-family: 'Helvetica Neue', sans-serif;
        text-align: center;
        border-bottom: 2px solid #2E86C1;
        padding-bottom: 10px;
    }
    .result-card {
        background-color: #E8F8F5;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1ABC9C;
        margin-top: 20px;
    }
    </style>
    <h1 class="main-header">🎓 Student Score Predictor</h1>
    <p style="text-align: center;">Enter the student's details below to predict their math score.</p>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('student_model.pkl')

model = load_model()

# Create the frontend form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["female", "male"])
        race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
        lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
        
    with col2:
        education = st.selectbox("Parental Education", [
            "some high school", "high school", "some college", 
            "associate's degree", "bachelor's degree", "master's degree"
        ])
        test_prep = st.selectbox("Test Preparation", ["none", "completed"])
        
    submitted = st.form_submit_button("Predict Score")

# Backend prediction logic
if submitted:
    # Format the input exactly how the model expects it
    input_data = pd.DataFrame({
        'gender': [gender],
        'race/ethnicity': [race],
        'parental level of education': [education],
        'lunch': [lunch],
        'test preparation course': [test_prep]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display result using custom HTML
    st.markdown(f"""
        <div class="result-card">
            <h3 style="color: #1ABC9C; margin-top: 0;">Predicted Math Score: {prediction:.1f} / 100</h3>
        </div>
    """, unsafe_allow_html=True)