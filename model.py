import eda
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

def logistic():
    return LogisticRegression()

def decision_tree():
    return DecisionTreeClassifier()

def naive_bayes():
    return GaussianNB()

def randomf():
    est = st.sidebar.slider('Select N_ESTIMATORS', min_value=1, max_value=1000, value=139)
    return RandomForestClassifier(n_estimators=est)

def suppot_vm():
    return SVC()

def KNneigbors():
    neigh = st.sidebar.slider('Select n_neighbors', min_value=1, max_value=50, value=10)
    return KNeighborsClassifier(n_neighbors=neigh)

def initialise_model(df, classifier_name, features, preprocessing):
    default = ['trestbps', 'chol', 'thalach', 'fbs', 'c_trestbps', 'c_chol', 'target']
    
    # If user want to drop more features along with default
    if 'default' in features and len(features) > 1:
        features.remove('default')
        for fet in features:
            default.append(fet)
        X = df.drop(default, axis=1)
    elif 'default' in features:
        X = df.drop(default, axis=1)
    else:
        X = df.drop(features, axis=1)
    
    Y = df['target']
    if preprocessing == 'MinMaxScaler':
        s = MinMaxScaler()
    else:
        s = StandardScaler()
    X = s.fit_transform(X)
    kfold = st.sidebar.slider('Select Kfold', min_value=2, max_value=20, value=5)
    
    models = {}
    for mod in classifier_name:
        if mod == 'Logistic Regression':
            lr = logistic()
            models['Logistic Regression'] = lr
        elif mod == 'Decision Tree':
            tree = decision_tree()
            models['Decision TREE'] = tree
        elif mod == 'Naive Bayes':
            nb = naive_bayes()
            models['Naive Bayes'] = nb
        elif mod == 'Random Forest':
            rdf = randomf()
            models['Random Forest'] = rdf
        elif mod == 'Support Vector Machine':
            svc = suppot_vm()
            models['Support Vector Machine'] = svc
        elif mod == 'KNN':
            knn = KNneigbors()
            models['KNN'] = knn
        elif mod == 'All':
            lr, tree, nb, rdf, svc, knn = logistic(), decision_tree(), naive_bayes(), randomf(),\
                suppot_vm(), KNneigbors()
            
            models = {'Logistic Regression':lr, 'Decision Tree': tree, \
                'Naive Bayes': nb, 'Random Forest': rdf, 'Support Vector Machine':svc, 'KNN': knn}

    with st.spinner('Building Model...!'):
        for j in models.keys():
            train_result = cross_val_score(models[j], X, Y, cv=kfold)
            st.text("{0}\n\tMean Accuracy: {1}\n\tMaximum Accuracy: {2}\n\tMinimum Accuracy: {3}".format(\
                j, np.mean(train_result)*100, train_result.max()*100, train_result.min()*100))
    

def model_param(df):
    st.markdown("""<style>.small-font {font-size:12px !important;}</style>""", unsafe_allow_html=True)

    classifier_name = st.multiselect('Select the model', ['All', 'Logistic Regression', 'Decision Tree',\
            'Naive Bayes', 'Random Forest', 'Support Vector Machine', 'KNN'])
    
    feat = list(df.drop('target', axis=1))
    feat.insert(0, 'default')
    features = st.multiselect('Features to be dropped', feat, default='default')
    features.append('target')

    st.markdown('<p class="small-font">Default features to be dropped: <b>trestbps, chol,\
        thalach, fbs, c_trestbps, c_chol</b></p>', unsafe_allow_html=True)
    st.markdown('<p class="small-font">Here, <b>chol, thalach, trestbps</b> are continuous values and\
        therefore have been converted into Categorical Values. The catgorical columns are, \
        <b>c_chol, c_thalach, c_trestbps </b> respectively.<br><br></p>', unsafe_allow_html=True)
    
    preprocessing = st.sidebar.selectbox('Select PreProcessing', ['StandardScaler', 'MinMaxScaler'],\
        index=0)
    st.sidebar.markdown('<p class="small-font">By default, <b>Standard Scaler</b> will be used.<br><br></p>',\
        unsafe_allow_html=True)
    
    if classifier_name and features is not None:
        initialise_model(df, classifier_name, features, preprocessing)

def main(data):
    model_param(data)