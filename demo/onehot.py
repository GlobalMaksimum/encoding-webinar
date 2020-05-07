import pandas as pd
import numpy as np 
import streamlit as st
from PIL import Image

def content():
    
    st.markdown("""
    ## Another Idea: One Hot Encoding
    1. Generate columns/attributes/features as many as the number of distrinct values in encoded column/attribute/feature
    2. Set only 1 relevant column/attribute/feature value to 1 and 0 others in the encoded domain
    """)
    
    image1 = Image.open('images/ohe2.png')
    st.image(image1, use_column_width=True)

    df = pd.read_csv('data/breast-cancer.csv')
    
    df.rename(columns={'Class': 'target'}, inplace=True)

    df.fillna('unknown', inplace = True)
    
    feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))
    X_ohe = pd.DataFrame(df[feat])
    
    st.dataframe(df[f'{feat}'])

    st.subheader("`sklearn` has a special module for One Hot Encoder:")
    
    with st.echo():
        from sklearn.preprocessing import OneHotEncoder
            
    showImplementation = st.checkbox('Show Code', key='Similarity1') 
    
    if showImplementation:
        with st.echo():
            ohe = OneHotEncoder()
            X_ohe = ohe.fit_transform(df[[feat]])

    st.subheader("In addition to `pandas`, `sklearn` has also a module 'get_dummies' for One Hot Encoding")
    
    showImplementation2 = st.checkbox('Show Code', key='Similarity2') 
    
    if showImplementation2:
        with st.echo():
            X_ohe = pd.get_dummies(df[feat])
         
    button2 = st.button('Apply One Hot Encoding')
    if button2:
        X_ohe = pd.get_dummies(df[feat])
        st.dataframe(X_ohe)
        
    st.markdown(""" 
    ### Why?
    - How similar/dissimilary each value in breast-quad with respect to each other ?
    - Note that the answer of question is mainly related with the encoding you use.
    """)

    st.markdown('### 1. Similarity/Dissimilarity each value in **breast-quad** wrt each other by LabelEncoder')
    with st.echo():
        df['breast-quad'].unique()
        
    st.markdown('Unique values are')
    st.write(df['breast-quad'].unique())
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    enc_le = le.fit(df['breast-quad']).transform(df['breast-quad'].unique()).reshape(-1,1)
    
    button = st.button('Apply Label Encoding for Unique Values')
    
    if button:
        st.subheader('Mapped Values')
        st.write(enc_le)
            
    
    st.info(":gem: Similarity/Dissimilarity is calculated by pairwise distance matrices")
    with st.echo():
        from sklearn.metrics import pairwise_distances
        
    st.markdown('#### Calculate Pairwise Distance Between Encoding of Each Unique Value')
    with st.echo():
        pairwise_distances(enc_le.reshape(-1,1))
    st.write(pairwise_distances(enc_le.reshape(-1,1)))



    st.markdown('### 2. Similarity/Dissimilarity each value in breast-quad wrt each other by our new encoding scheme')
                                   
    ohe = OneHotEncoder()
    
    with st.echo():
        enc_ohe = ohe.fit(df[['breast-quad']]).transform(df['breast-quad'].unique().reshape(-1,1)).toarray()
    st.write(enc_ohe)

    st.subheader('Calculate Pairwise Distance Between Each Unique Value')
    with st.echo():
        pairwise_distances(enc_ohe)

    st.write(pairwise_distances(enc_ohe))

    st.markdown('Hence our new encoding preserves relative similarity/dissimilarity of each unique value.')
