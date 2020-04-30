import pandas as pd
import numpy as np 
import streamlit as st

def content():
    st.markdown("""
    ## Another Idea
    1. Generate columns/attributes/features as many as the number of distrinct values in encoded column/attribute/feature
    2. Set only 1 relevant column/attribute/feature value to 1 and 0 others in the encoded domain
    """)

    df = pd.read_csv('data/breast-cancer.csv')

    feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

    st.dataframe(df[f'{feat}'])

    with st.echo():
        pd.get_dummies(df[feat])

    st.write(pd.get_dummies(df[feat]))

    st.markdown(""" 
    ### Why?
    - How similar/dissimilary each value in breast-quad with respect to each other ?
    - Note that the answer of question is mainly related with the encoding you use.
    """)

    st.markdown('### 1. Similarity/Dissimilarity each value in **breast-quad** wrt each other by LabelEncoder')
    with st.echo():
        df['breast-quad'] =  df['breast-quad'].fillna('unknown')
        unq_values = df['breast-quad'].unique()
        unq_values

    with st.echo():
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        enc_le = le.fit(df['breast-quad']).transform(unq_values).reshape(-1,1)
    st.write(enc_le)

    st.markdown('#### Calculate Pairwise Distance Between Encoding of Each Unique Value')
    with st.echo():
        from  sklearn.metrics import pairwise_distances
        pairwise_distances(enc_le.reshape(-1,1))
    st.write(pairwise_distances(enc_le.reshape(-1,1)))



    st.markdown('### 2. Similarity/Dissimilarity each value in breast-quad wrt each other by our new encoding scheme')
    with st.echo():
        from sklearn.preprocessing import OneHotEncoder
        ohe= OneHotEncoder()
        enc_ohe = ohe.fit(df[['breast-quad']]).transform(unq_values.reshape(-1,1)).toarray()
        enc_ohe

    st.subheader('Calculate Pairwise Distance Between Each Unique Value')
    with st.echo():
        pairwise_distances(enc_ohe)

    st.write(pairwise_distances(enc_ohe))

    st.error('3D VIS WILL COME HERE')

    st.markdown('Hence our new encoding preserves relative similarity/dissimilarity of each unique value.')