import streamlit as st
import model
import eda
import predictor


st.set_page_config(layout="wide")
st.title('Heart Disease Predictor')
st.write("""Predicitng whether the user has Heart Disease or Not""")

st.sidebar.title('Select Action')
action = st.sidebar.radio('Options: ', ['EDA & Data PreProcessing', 'Model', 'Prediction'])

data = eda.read_data()
data = eda.rearrange_dataframe(data)

if action == 'EDA & Data PreProcessing':
    eda.main(data)
elif action == 'Model':
    model.main(data)
elif action == 'Prediction':
    predictor.main(data)