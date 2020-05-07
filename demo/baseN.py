import streamlit as st
import pandas as pd
import numpy as np 

def content():

    df = pd.read_csv('data/breast-cancer.csv')
    
    df.rename(columns={'Class': 'target'}, inplace=True)
    df.fillna('unknown', inplace = True)
    
    st.header("BaseN Encoding")

    #st.header("We Can Reduce Dimensionality Even More by Applying BaseN Encoding")

    st.markdown("BaseN encoding converts a category into base N digits.")

    st.markdown("By reducing dimensionality, we also lose some information.")

    st.markdown("Here we apply base 3 encoding for ordinal columns.")

    feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

    import category_encoders as ce
    
    X_baseN = pd.DataFrame(df[feat])

    st.dataframe(X_baseN)

    #X_baseN[['deg-malig']] = X_baseN[['deg-malig']].astype('object')


    baseNEnc = ce.BaseNEncoder(base = 3)
    X_baseN = baseNEnc.fit_transform(X_baseN)

    button = st.button('Apply BaseN Encoding')
    if button:
        st.dataframe(X_baseN)