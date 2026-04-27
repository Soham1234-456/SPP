import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# -------------------------------
# Load Files
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))


data = pd.read_csv("student._performance.full.csv")

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="ML Dashboard", layout="wide")

st.title("📊 Student Performance ML Dashboard")

# -------------------------------
# Sidebar Navigation
# -------------------------------
menu = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Prediction", "Model Comparison"]
)

# -------------------------------
# DASHBOARD PAGE
# -------------------------------
if menu == "Dashboard":

    st.header("📊 Dataset Overview")

    st.write(data)

    st.subheader("📈 Data Visualization")

    import matplotlib.pyplot as plt
    import seaborn as sns

    st.header("📊 Data Visualization")

# Select only numeric columns
    numeric_data = data.select_dtypes(include=['number'])

# -------------------------------
# 1. Histogram (Score Distribution)
# -------------------------------
    st.subheader("📊 Score Distribution")

    fig1 = plt.figure()
    sns.histplot(data["Previous_Scores"], bins=30)
    plt.xlabel("Score")
    plt.ylabel("Frequency")

    st.pyplot(fig1)

# -------------------------------
# 2. Scatter Plot (Hours vs Score)
# -------------------------------
    st.subheader("📈 Study Hours vs Score")

    fig2 = plt.figure()
    sns.scatterplot(x="Hours_Studied", y="Previous_Scores", data=data)

    st.pyplot(fig2)

# -------------------------------
# 3. Box Plot (Outliers)
# -------------------------------
    st.subheader("📦 Attendance vs Score")

    fig3 = plt.figure()
    sns.boxplot(x=data["Attendence"])

    st.pyplot(fig3)

# -------------------------------
# 4. Correlation Heatmap
# -------------------------------
    st.subheader("🔥 Correlation Heatmap")

    corr = numeric_data.corr()

    fig4 = plt.figure()
    sns.heatmap(corr, annot=True)

    st.pyplot(fig4)

   
# -------------------------------
elif menu == "Prediction":

    st.header("🤖 Predict Student Score")

    col1, col2, col3 = st.columns(3)

    with col1:
        hours = st.slider("Hour Studied", 0, 12, 4)

    with col2:
        prev_score = st.slider("Previous Score", 0, 100, 50)

    with col3:
        sleep = st.slider("Sleep Hours", 0, 12, 6)

    col4, col5, col6 = st.columns(3)

    with col4:
        extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])

    with col5:
        sample_papers = st.slider("Sample Papers Practiced", 0, 20, 5)

    with col6:
        attendance = st.slider("Attendance (%)", 0, 100, 75)

    # categorical inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    school_type = st.selectbox("School Type", ["Government", "Private"])

    if st.button("Predict"):

        # -------------------------------
        # Convert categorical to numeric
        # -------------------------------

        extracurricular_val = 1 if extracurricular == "Yes" else 0
        gender_val = 1 if gender == "Male" else 0
        school_val = 1 if school_type == "Private" else 0

        # -------------------------------
        # FINAL INPUT (8 FEATURES)
        # -------------------------------
        input_data = np.array([[
            hours,
            prev_score,
            extracurricular_val,
            sleep,
            sample_papers,
            attendance,
            gender_val,
            school_val
        ]])

        
        # Predict
        prediction = model.predict(input_data)

        st.subheader(f"📈 Predicted Score: {prediction[0]:.2f}")

        # Feedback
        if prediction[0] >= 75:
            st.success("🔥 Excellent")
        elif prediction[0] >= 40:
            st.info("👍 Good")
        else:
            st.warning("⚠️ Need more effort")
# -------------------------------
# MODEL COMPARISON PAGE
# -------------------------------
elif menu == "Model Comparison":

    st.header("🏆 Model Comparison")

    # Example values (replace with real if saved)
    models = ["Linear Regression", "Decision Tree", "Random Forest"]
    scores = [0.95, 0.90, 0.97]

    fig, ax = plt.subplots()
    ax.bar(models, scores)
    ax.set_xlabel("Models")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Performance")

    st.pyplot(fig)

    st.write("Best Model: Random Forest")