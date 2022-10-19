from re import X
import streamlit as st
import numpy as np
import pickle

from yaml import load

def Load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data



data = Load_model()

regressor_loaded = data["model"]
le_country = data["le_country"]
le_ed = data["le_ed"]


def show_predict_page():

    st.title("Data Scientists Salary Prediction App")

    st.write("""
    ### We need some information to predict the Salary
    """)


    countries=('United States of America', 'Germany', 'United Kingdom of Great Britain and Northern Ireland',
           'India', 'Canada', 'France', 'Brazil', 'Spain', 'Netherlands', 'Australia', 'Italy', 'Italy', 
           'Italy', 'Russian Federation', 'Switzerland'
           )

    education=('Master’s degree', 'Bachelor’s degree', 'Less than a Bachelors',
       'Post grad'
       )

    country= st.selectbox("Country", countries)
    education= st.selectbox("Education Level", education)
    experience= st.slider("Years of Experince", 0, 50, 3)
    
    ok= st.button("Claculate Salary")
    if ok:
        x=np.array([[country, education, experience]])
        x[:, 0]=le_country.transform(x[:, 0])
        x[:, 1] = le_ed.transform(x[:, 1])
        x= x.astype(float)

        salary= regressor_loaded.predict(x)
        st.subheader(f"The Estimated Salary is ${salary[0]:.2f}")

