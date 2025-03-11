import os
import pickle
import streamlit as st

st.set_page_config(page_title="Disease Prediction System", layout="wide", page_icon="🧑‍⚕️")

# Load saved models
def load_model(filename):
    working_dir = os.path.dirname(os.path.abspath(__file__))
    return pickle.load(open(os.path.join(working_dir, f"model_saved/{filename}"), 'rb'))


diabetes_model = load_model("diabetes_model.sav")
heart_disease_model = load_model("heart_model.sav")
parkinsons_model = load_model("parkinson_model.sav")

# Prediction function
def predict_disease(model, user_inputs):
    try:
        user_inputs = [float(x) for x in user_inputs]
        prediction = model.predict([user_inputs])
        return "✅ Positive" if prediction[0] == 1 else "❌ Negative"
    except ValueError:
        return "⚠️ Invalid input! Please enter numbers."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Create UI Tabs
tab1, tab2, tab3 = st.tabs(["🩸 Diabetes Prediction", "❤️ Heart Disease Prediction", "🧠 Parkinson's Prediction"])

# Diabetes Tab
with tab1:
    st.title("Diabetes Prediction using ML")
    st.info("""
    **Diabetes** is a chronic disease that occurs when the pancreas is no longer able to make insulin, 
    or when the body cannot effectively use the insulin it produces.
    """)
    
    col1, col2, col3 = st.columns(3)
    Pregnancies = col1.number_input('Number of Pregnancies', min_value=0, step=1, value=0)
    Glucose = col2.number_input('Glucose Level', min_value=0.0, step=0.1, value=80.0)
    BloodPressure = col3.number_input('Blood Pressure', min_value=0.0, step=0.1, value=70.0)
    SkinThickness = col1.number_input('Skin Thickness', min_value=0.0, step=0.1, value=20.0)
    Insulin = col2.number_input('Insulin Level', min_value=0.0, step=0.1, value=79.0)
    BMI = col3.number_input('BMI', min_value=0.0, step=0.1, value=24.0)
    DiabetesPedigreeFunction = col1.number_input('Diabetes Pedigree Function', min_value=0.0, step=0.01, value=0.5)
    Age = col2.number_input('Age', min_value=1, step=1, value=25)

    if st.button('Diabetes Test Result'):
        result = predict_disease(diabetes_model, [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        st.success(result)

# Heart Disease Tab
with tab2:
    st.title("Heart Disease Prediction using ML")
    st.info("""
    **Heart disease** refers to various types of heart conditions, including coronary artery disease, heart attacks, and more.
    """)
    
    col1, col2, col3 = st.columns(3)
    age = col1.number_input('Age', min_value=1, step=1, value=50)
    sex = col2.radio('Sex', [0, 1])
    cp = col3.selectbox('Chest Pain Type', [0, 1, 2, 3])
    trestbps = col1.number_input('Resting Blood Pressure', min_value=0, step=1, value=120)
    chol = col2.number_input('Serum Cholesterol', min_value=0, step=1, value=200)
    fbs = col3.radio('Fasting Blood Sugar > 120 mg/dl', [0, 1])
    restecg = col1.selectbox('Resting ECG Results', [0, 1, 2])
    thalach = col2.number_input('Max Heart Rate', min_value=0, step=1, value=150)
    exang = col3.radio('Exercise Induced Angina', [0, 1])
    oldpeak = col1.number_input('ST Depression', min_value=0.0, step=0.1, value=1.0)
    slope = col2.selectbox('Slope of ST Segment', [0, 1, 2])
    ca = col3.number_input('Major Vessels Colored', min_value=0, step=1, value=0)
    thal = col1.selectbox('Thalassemia Type', [0, 1, 2])

    if st.button('Heart Disease Test Result'):
        result = predict_disease(heart_disease_model, [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        st.success(result)

    st.info(
    '''* Age: Patient’s Age in years.\n'''
    '''* Sex: Patient’s Gender. (M = Male, F = Female)\n'''
    '''* ChestPainType: Chest Pain type. (4 values: ATA, NAP, ASY, TA)\n'''
    '''* RestingBP: resting Blood Pressure. ( in mm Hg )\n'''
    '''* Cholesterol: Serum Cholesterol. ( in mg/dl )\n'''
    '''* FastingBS: Fasting Blood Sugar > 120 mg/dl. (0 = True, 1 = False)\n'''
    '''* RestingECG: resting Electroencephalographic result. (values: Normal, ST, LVH)\n'''
    '''* MaxHR: Maximum Heart Rate achieved.\n'''
    '''* ExerciseAngina: Exercise induced Angina. (N = No, Y = Yes)\n'''
    '''* Oldpeak: ST Depression induced by Exercise relative to rest.\n'''
    '''* ST_Slope: Slope of the peak exercise ST segment. (values: Up, Flat, Down)\n'''
    '''* HeartDisease:: Heart Disease occured. (0 = No, 1 = Yes)\n'''
    )

# Parkinson's Disease Tab
with tab3:
    st.title("Parkinson's Disease Prediction using ML")
    st.info("""
    **Parkinson’s disease** is a progressive nervous system disorder that affects movement.
    It is associated with tremors, stiffness, and slow movement.
    """)
    
    col1, col2, col3 = st.columns(3)
    MDVP_Fo = col1.number_input('MDVP:Fo(Hz) - Avg. Vocal Frequency', min_value=0.0, step=0.1, value=120.0)
    MDVP_Fhi = col2.number_input('MDVP:Fhi(Hz) - Max Vocal Frequency', min_value=0.0, step=0.1, value=130.0)
    MDVP_Flo = col3.number_input('MDVP:Flo(Hz) - Min Vocal Frequency', min_value=0.0, step=0.1, value=110.0)
    MDVP_Jitter = col1.number_input('MDVP:Jitter(%)', min_value=0.0, step=0.001, value=0.005)
    MDVP_Shimmer = col2.number_input('MDVP:Shimmer', min_value=0.0, step=0.001, value=0.03)
    NHR = col3.number_input('NHR - Noise to Tonal Ratio', min_value=0.0, step=0.01, value=0.02)
    HNR = col1.number_input('HNR - Harmonic Noise Ratio', min_value=0.0, step=0.1, value=20.0)
    RPDE = col2.number_input('RPDE - Dynamical Complexity', min_value=0.0, step=0.001, value=0.4)
    D2 = col3.number_input('D2 - Dynamical Complexity', min_value=0.0, step=0.1, value=2.0)

    if st.button("Parkinson's Test Result"):
        result = predict_disease(parkinsons_model, [MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_Shimmer, NHR, HNR, RPDE, D2])
        st.success(result)