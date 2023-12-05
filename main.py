import numpy as np
import joblib
import streamlit as st

# Loading the saved model
loaded_model = joblib.load('C:/salman\ML/kidny stone/model.joblib')

# Creating a function for prediction
def kidney_disease_prediction(input_data):
    # Changing the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The patient is not diagnosed with kidney disease'
    else:
        return 'The patient is diagnosed with kidney disease'

def main():
    # Giving a title
    st.title('Kidney Disease Prediction Web App')

    # Getting the input data from the user
    gravity = st.text_input('Gravity')
    ph = st.text_input('pH')
    osmo = st.text_input('Osmo')
    cond = st.text_input('Cond')
    urea = st.text_input('Urea')
    calc = st.text_input('Calc')

    # Code for prediction
    diagnosis = ''

    # Creating a button for prediction
    if st.button('Kidney Disease Test Result'):
        input_params = [gravity, ph, osmo, cond, urea, calc]
        diagnosis = kidney_disease_prediction(input_params)

    st.success(diagnosis)

if __name__ == '__main__':
    main()

