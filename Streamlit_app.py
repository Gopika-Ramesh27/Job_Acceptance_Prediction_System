import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Candidate Placement Predictor", page_icon="💼", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load('placement_model.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, columns

model, model_columns = load_model()

st.title("💼 Candidate Placement Predictor")
st.write("Predict if a candidate will accept a job offer based on their interview and background details.")
st.write("---")

st.sidebar.header("Candidate Details")

technical_score = st.sidebar.slider("Technical Score", 0, 100, 75)
years_of_experience = st.sidebar.slider("Years of Experience", 0, 20, 2)
degree_percentage = st.sidebar.slider("Degree Percentage", 0, 100, 75)
skills_match_percentage = st.sidebar.slider("Skills Match %", 0, 100, 80)
previous_ctc_lpa = st.sidebar.slider("Previous CTC (LPA)", 0.0, 30.0, 5.0, 0.5)
expected_ctc_lpa = st.sidebar.slider("Expected CTC (LPA)", 0.0, 40.0, 7.5, 0.5)

avg_interview_score = technical_score
if years_of_experience == 0:
    experience_category = 'Fresher'
elif years_of_experience <= 2:
    experience_category = 'Junior'
else:
    experience_category = 'Senior'

academic_band = 'Low' if degree_percentage < 65 else ('Average' if degree_percentage <= 80 else 'High')
skills_match_level = 'Low' if skills_match_percentage < 60 else ('Medium' if skills_match_percentage <= 80 else 'High')
interview_eval = 'Needs Improvement' if technical_score < 50 else ('Good' if technical_score <= 75 else 'Excellent')

col1, col2 = st.columns(2)

with col1:
    st.subheader("Candidate Overview")
    st.write(f"**Average Score:** {avg_interview_score}%")
    st.write(f"**Experience:** {experience_category}")
    st.write(f"**Academic Band:** {academic_band}")
    st.write(f"**Skills Match:** {skills_match_level}")
    st.write(f"**Evaluation:** {interview_eval}")

with col2:
    st.subheader("Prediction Result")
    
    if st.button("Predict Placement Status", type="primary", use_container_width=True):
        # Format input row
        raw_input = {
            'years_of_experience': years_of_experience,
            'technical_score': technical_score,
            'aptitude_score': technical_score,
            'communication_score': technical_score,
            'degree_percentage': degree_percentage,
            'skills_match_percentage': skills_match_percentage,
            'previous_ctc_lpa': previous_ctc_lpa,
            'expected_ctc_lpa': expected_ctc_lpa,
            'avg_interview_score': avg_interview_score,
            'experience_category': experience_category,
            'academic_band': academic_band,
            'skills_match_level': skills_match_level
        }

        input_df = pd.get_dummies(pd.DataFrame([raw_input]), drop_first=True)
        input_ready = input_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_ready)[0]
        prob = model.predict_proba(input_ready)[0][1] * 100

        if prediction == 1:
            st.success(f"### 🎉 Placed")
            st.write(f"**Probability:** {prob:.2f}%")
        else:
            st.error(f"### ❌ Not Placed")
            st.write(f"**Probability:** {prob:.2f}%")