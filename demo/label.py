import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

def content():

    df = pd.read_csv('data/breast-cancer.csv')

    st.header('How to Generalize Dictionary Encoding into N Distinct Values?')

    feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

    st.dataframe(df[f'{feat}'])
        
    with st.echo():
        df[feat].unique()
    
    st.markdown('Unique values are')
    st.write(df[feat].unique())
    
    button = st.button('Fill Missing Values')
    
    if button:
        with st.echo():
            df[feat] = df[feat].fillna('unknown')
            
    df[feat] = df[feat].fillna('unknown')
    
    st.subheader('`sklearn` has a great module for this: Label Encoder')

    with st.echo():
        from sklearn.preprocessing import LabelEncoder  
        
    image1 = Image.open('images/label.png')

    st.image(image1, use_column_width=True)
    
    X_label = pd.DataFrame(df[feat])
    
    showImplementation = st.checkbox('Show Code', key='Similarity1') 
    
    if showImplementation:
        with st.echo():
            le = LabelEncoder()
            X_label[f'tranformed_{feat}'] = le.fit_transform(df[feat])
    
    le = LabelEncoder()
    X_label[f'tranformed_{feat}'] = le.fit_transform(df[feat])
    
    button2 = st.button('Apply Label Encoding')
    if button2:
        st.subheader('Mapped Values')
        st.dataframe(X_label)
        
    st.warning("""
     :exclamation: What are the potential issues with Label Encoder approach?
    """)