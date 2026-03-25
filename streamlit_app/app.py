import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------------------
# Load trained model
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)

st.title("❤️ Heart Disease Type Prediction")
st.write(
    "This application predicts the **type of heart disease** based on patient "
    "symptoms, medical history, and clinical measurements."
)

st.markdown("---")

# ---------------------------------
# User Inputs
# ---------------------------------

Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.number_input("Age", min_value=18, max_value=100, value=50)

Hypertension = st.selectbox("Hypertension", [0, 1])
Heart_Disease = st.selectbox("Previous Heart Disease", [0, 1])

Family_History = st.selectbox("Family History", ["Yes", "No"])
Heart_Procedure = st.selectbox("Heart Procedure", ["Yes", "No"])

Chest_pain = st.selectbox("Chest Pain", [0, 1])
Max_Heart_Rate = st.number_input("Max Heart Rate", 60, 220, 150)

Avg_Glucose_level = st.number_input(
    "Average Glucose Level", min_value=50.0, max_value=300.0, value=120.0
)

Smoking_Status = st.selectbox(
    "Smoking Status", ["Never", "Former", "Current"]
)

Cholesterol = st.number_input(
    "Cholesterol Level", min_value=100, max_value=600, value=200
)

Shivering = st.selectbox("Shivering / Dizziness", ["Yes", "No"])
Palpitation = st.selectbox("Palpitation", ["Yes", "No"])
Pain = st.selectbox("Pain", ["Yes", "No"])

Body_Part = st.selectbox(
    "Body Part Affected", ["Chest", "Arm", "Back", "Neck"]
)

Nose_Bleeding = st.selectbox("Nose Bleeding", ["Yes", "No"])
Vomiting = st.selectbox("Vomiting", ["Yes", "No"])

Serum_Creatinine = st.number_input(
    "Serum Creatinine", min_value=0.5, max_value=5.0, value=1.0
)

Serum_Sodium = st.number_input(
    "Serum Sodium", min_value=120, max_value=160, value=135
)

Insomnia = st.selectbox("Insomnia", ["Yes", "No"])
breath_fatigue = st.selectbox("Breath Fatigue", [0, 1])

Memoryloss = st.selectbox("Memory Loss", ["Yes", "No"])
Exercise_stress = st.selectbox("Exercise Stress", ["Yes", "No"])
Discoloration = st.selectbox("Discoloration", ["Yes", "No"])
Loss_of_Appetite = st.selectbox("Loss of Appetite", ["Yes", "No"])

st.markdown("---")

# ---------------------------------
# Prediction
# ---------------------------------

if st.button("🔍 Predict Heart Disease Type"):
    
    input_df = pd.DataFrame([{
    "id": 0,  # dummy ID
    "Gender": Gender,
    "Age": Age,
    "Hypertension": Hypertension,
    "Heart Disease": Heart_Disease,
    "Family History": Family_History,
    "Heart Procedure": Heart_Procedure,
    "Chest pain": Chest_pain,
    "Max Heart Rate": Max_Heart_Rate,
    "Avg Glusoce level": Avg_Glucose_level,
    "Smoking Status": Smoking_Status,
    "Cholesterol": Cholesterol,
    "Swelling": "No",  # dropped feature → default
    "Shivering/dizziness": Shivering,
    "Palpitation": Palpitation,
    "Pain": Pain,
    "Body Part": Body_Part,
    "Body Parts": Body_Part,  # duplicate column
    "Nose Bleeding": Nose_Bleeding,
    "Vomiting": Vomiting,
    "Serum_Creatinine": Serum_Creatinine,
    "Serum_Sodium": Serum_Sodium,
    "Insomnia": Insomnia,
    " breath fatigue": breath_fatigue,
    "Memoryloss": Memoryloss,
    "Execise stress": Exercise_stress,
    "Discoloration": Discoloration,
    "Loss of Appetite": Loss_of_Appetite
}])


    prediction = model.predict(input_df)[0]

    st.success(
        f"🩺 **Predicted Heart Disease Type:** {prediction}"
    )
