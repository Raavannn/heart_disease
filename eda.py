import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

def read_data():
    return pd.read_csv('heart.csv')

def impute_bp(bp):
    if bp <= 120:
        return 0
    else:
        return 1

def impute_chol(chol):
    if 125 <= chol <= 200:
        return 0
    elif chol > 200:
        return 1
    else:
        return 2

def impute_thalach(col):
    a_rate = col[1]
    m_rate = 220 - col[0]
    lt_rate = (m_rate * 60)/100
    ht_rate = (m_rate * 80)/100
    
    if lt_rate <= a_rate <= ht_rate:
        return 0 #Normal
    elif a_rate < lt_rate:
        return 1 #Low
    elif a_rate > ht_rate:
        return 2

def preprocessing(df):
    st.dataframe(df.drop(['c_thalach', 'c_chol', 'c_trestbps'], axis=1).head(5))
    st.write('Features **trestbps**, **chol** & **thalach** have continuous values, these values can be conerted into\
    categorical values on the basis of following information.')
    st.text("Trestbps\n\tif < 120 then normal, if > 120 then High")
    st.text("Cholestrol\n\tif cholestrol levels less than 200 then Normal, else High")
    st.text("Thalach: Maximum Heart rate achieved\n\t0: Ideal/Normal(60-80)% of 220 - age\n\
        1: Low (less than 60%)\n\t2: High (greater than 80%)")
    
    st.write("""RestECG has been categorized as Normal (0), ST-T wave Abnormality\n \
    (1) & Left Ventricular Hypertrophy (2) The number of patients categorized as 2 are \n\
    relatively very small as compared to other two. Thus treating 2 as outlier & removing it. \n\
    No Information was given for thal == 0, hence dropping it.""")    

def rearrange_dataframe(df):
    df['c_thalach'] = df[['age', 'thalach']].apply(impute_thalach, axis=1)
    df['c_chol'] = df['chol'].apply(impute_chol)
    df['c_trestbps'] = df['trestbps'].apply(impute_bp)
    
    df.drop(df[df['restecg'] == 2].index, inplace=True)
    df.drop(df[df['thal'] == 0].index, inplace=True)
    col = list(df.columns.values)
    col.remove('target')
    col.append('target')
    return df.reindex(columns=col)

def dist_plots(df):
    fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(15,7))
    index = 0
    axs = axes.flatten()
    for col in df.drop(['c_thalach', 'c_chol', 'c_trestbps'], axis=1).columns[:14]:
        sns.histplot(df[col], kde=True, ax=axs[index], palette='cividis')
        index += 1
    plt.tight_layout(pad=0.5, h_pad=3.0, w_pad=1.5)
    st.pyplot(fig)

def box_plots(df):
    fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(15,7))
    index = 0
    axs = axes.flatten()
    for col in df.drop(['c_thalach', 'c_chol', 'c_trestbps'], axis=1).columns[:14]:
        sns.boxplot(y=col, data=df, ax=axs[index], palette='cividis')
        index += 1
    plt.tight_layout(pad=0.5, h_pad=3.0, w_pad=1.5)
    st.pyplot(fig)

def heatmap(df):
    fig = plt.figure(figsize=(18,8))
    sns.heatmap(df.corr(), annot=True, cmap='viridis')
    st.pyplot(fig)

def load_plots(data):
    st.write("Let's checkout data distribution")
    with st.spinner(text='Generating Plot'):
        dist_plots(data)

    st.write("Box Plot")
    with st.spinner(text='Generating Plot'):
        box_plots(data)

    st.write('Heatmap of the dataset')
    with st.spinner(text='Generating Plot'):
        heatmap(data)
        st.success('Done')

    st.write("\nFrom the Heatmap, we can infer that features **trestbps**, **fbs** & **chol** shows \
        very low correlation with the target variable. Hence, we should drop them while building model.")

def main(data):
    preprocessing(data)
    if st.button('Load Plots'):
        load_plots(data)