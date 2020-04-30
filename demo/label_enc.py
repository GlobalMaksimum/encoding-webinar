import streamlit as st
import pandas as pd
import numpy as np


def content():

    df = pd.read_csv('data/breast-cancer.csv')


    st.header('How to Generalize Dictionary Encoding into N distrinct values ?')

    feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

    st.dataframe(df[f'{feat}'])

    with st.echo():
        df[feat] =  df[feat].fillna('unknown')
        df[feat].unique()

    st.markdown('Unique values are')
    st.write(df[feat].unique())



    st.subheader('`sklearn` has a great module for this')

    with st.echo():
        from sklearn.preprocessing import LabelEncoder  
        
    with st.echo():
        le = LabelEncoder()

    button = st.button('fit_transform')

    if button:
        with st.echo():
            le.fit_transform(df[feat])
        st.write(le.fit_transform(df[feat]))

    st.markdown("""
    ### Write into Q & A
    What are the potential issues with LabelEncoder approach ?
    """)