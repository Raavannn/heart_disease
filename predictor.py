import eda
import model
import streamlit as st
import pandas as pd
import numpy as np
import sklearn.metrics as mt
from sklearn.model_selection import train_test_split

def predict(input, df):
    X = df.drop(['trestbps', 'chol', 'thalach', 'fbs', 'c_trestbps', 'c_chol', 'target'], axis=1)
    Y = df['target']
    s = model.StandardScaler()
    X = s.fit_transform(X)
    
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.3, random_state=34)
    knn = model.KNneigbors()
    knn.fit(X_TRAIN, Y_TRAIN)
    pred = knn.predict(X_TEST)

    cm = mt.confusion_matrix(Y_TEST, pred)
    TP = cm[0][0]
    TN = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]
    accuracy = ((TP + TN)/(TP + TN + FN + FP))

    st.write('Prediciting Outcome with an accuracy of: {:.2f} %'.format(accuracy*100))

    # Training Testing of Model us Done, now let's Evaluate
    input.drop(['trestbps', 'chol', 'thalach', 'fbs', 'c_trestbps', 'c_chol'], axis=1, inplace=True)
    test = np.array(input.loc[0].to_list()).reshape(1, -1)
    result = knn.predict(test)
    st.header('RESULT')
    if result[0] == 1:
        st.subheader('The patient is diagnosed with Heart Disease')
    else:
        st.subheader('The Patient is free from Heart Disease.')


def input_params():
    with st.expander("See explanation"):
        col1, col2 = st.columns(2)
        
        #Column 1
        col1.text("Sex:\n\tFemale - 0\n\tMale - 1")
        col1.text("Chest Pain:\n\tTypical Angina - 0\n\tAtypical Angina - 1\n\tNon-Anginal Pain - 2\
        \n\tAsymptomatic - 3")
        col1.text("Trestbps:\n\tNormal - 0 (<120)\n\tHigh - 1 (>120)")
        col1.text("FBS (> 120):\n\tFalse - 0\n\tTrue - 1")
        
        #Column 2
        col2.text("Restecg:\n\tNormal ECG - 0\n\tHaving ST-T Wave abnormality - 1\
            \n\tLeft Ventricule Hypertrophy - 2")
        col2.text("Exang:\n\tNo - 0\n\tYes - 1")
        col2.text("Slope:\n\tUnslopping - 0\n\tFlat - 1\n\tDownslopping - 2")
        col2.text("Thal:\n\tNormal Thalasemmia - 1\n\tFixed Thalasemmia - 2\n\tReversable Defect Thalasemmia - 3")

    params = {}
    params['age'] = st.text_input('Enter your Age: ')
    params['sex'] = st.selectbox('Sex ', ['Female', 'Male'])
    params['cp'] = st.selectbox('Chest Pain, (cp)', ['Typical Angina', 'Atypical Angina', \
        'Non-Anginal Pain', 'Asymptomatic'])
    params['trestbps'] = st.text_input('Resting Blood Pressure, (trestbps)')
    params['chol'] = st.text_input('Cholestrol Level, (chol)')
    params['fbs'] = st.text_input('Fasting Blood Sugar, (fbs)')
    params['restecg'] = st.selectbox('Resting ECG, (restecg)', ['Normal ECG', 'Having ST-T Wave abnormality',\
         'Left Ventricule Hypertrophy'])
    params['thalach'] = st.text_input('Maximum Heart Rate achieved, (thalach)')
    params['exang'] = st.selectbox('Exercise Induced Angina, (exang)', ['No', 'Yes'])
    params['oldpeak'] = st.text_input('ST depression induced by exercise relative to rest, (oldpeak)')
    params['slope'] = st.selectbox('ST Segment, (Slope)', ['Unslopping', 'Flat', 'Downslopping'])
    params['ca'] = st.selectbox('Number of Major Vessels coloured by fluroscopy, (ca)', [0, 1, 2, 3, 4])
    params['thal'] = st.selectbox('Thalasemmia, (thal)', ['Normal Thalasemmia', 'Fixed Thalasemmia',\
         'Reversable Defect Thalasemmia'])

    return params

def input_preprocessing(params, df):
    mapping = {'Male': 1, 'Female': 0, 'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2,\
    'Asymptomatic': 3, 'Normal ECG': 0, 'Having ST-T Wave abnormality': 1, 'Left Ventricule Hypertrophy': 2,\
        'Yes': 1, 'No': 0, 'Unslopping': 0, 'Flat': 1, 'Downslopping': 2, 'Normal Thalasemmia': 1,\
            'Fixed Thalasemmia': 2, 'Reversable Defect Thalasemmia': 3}
    
    try:
        for j in ['age', 'trestbps', 'chol', 'fbs', 'thalach', 'oldpeak']:
            params[j] = float(params[j])
        
        for k,v in params.items():
            if v not in mapping.keys():
                params[k] = v
            else:
                params[k] = mapping[v]

        inp = pd.DataFrame.from_dict(params, orient='index').T
        inp['c_thalach'] = inp[['age', 'thalach']].apply(eda.impute_thalach, axis=1)
        inp['c_chol'] = inp['chol'].apply(eda.impute_chol)
        inp['c_trestbps'] = inp['trestbps'].apply(eda.impute_bp)

        with st.spinner(text='Predicting...!'):
            predict(inp, df)
    except Exception as e:
            st.write('Some Error occured: ', e)

def main(df):
    params = input_params()
    start = st.button('Predict')
    if start:
        input_preprocessing(params, df)