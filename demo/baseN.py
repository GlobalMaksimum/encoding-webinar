import streamlit as st
import pandas as pd
import numpy as np 

def content():

    df = pd.read_csv('data/breast-cancer.csv')
    
    df.rename(columns={'Class': 'target'}, inplace=True)
    df.fillna('unknown', inplace = True)
    df['deg-malig'] = df['deg-malig'].astype('object')
    
    st.title("BaseN Encoding")

    st.markdown(" * BaseN encoding converts a category into base N digits.")

    st.warning(":exclamation: By reducing dimensionality, we also lose some information.")

    feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))
    

    st.dataframe(df[feat])
    X_base_all = df[feat]
    
    #st.markdown("### Enter a number N for Base N")
    N = int(st.text_input('Enter a number N for Base N'))
    
    st.subheader("`category_encoders` has a module for BaseN Encoding")
    
    with st.echo():
        import category_encoders as ce
        baseNEnc = ce.BaseNEncoder(base = N)
    
    showImplementation = st.checkbox('Show Code', key='key1') 
    
    if showImplementation:
        with st.echo():
            X_baseN = baseNEnc.fit_transform(df[feat])

    X_baseN = baseNEnc.fit_transform(df[feat])
    X_base_all = pd.concat([X_base_all,X_baseN], axis=1)
    
    button = st.button('Apply BaseN Encoding')
    if button:
        st.dataframe(X_base_all)