import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("titanic_model.pkl", "rb"))

st.title("Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouse", 0, 10, 0)
parch = st.number_input("Parents/Children", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

sex_value = 0 if sex == "Male" else 1
embarked_value = {"S": 0, "C": 1, "Q": 2}[embarked]

if st.button("Predict"):

    input_data = pd.DataFrame(
        [[pclass, sex_value, age, sibsp, parch, fare, embarked_value]],
        columns=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
    )

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Not Survived")
