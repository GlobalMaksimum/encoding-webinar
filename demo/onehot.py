import pandas as pd
import numpy as np 
import streamlit as st
from PIL import Image

def content():
    
    st.markdown("""
    # Another Idea: One Hot Encoding
    * Generate columns/attributes/features as many as the number of distrinct values in encoded column/attribute/feature
    * Set only 1 relevant column/attribute/feature value to 1 and 0 others in the encoded domain
    """)
    
    image1 = Image.open('images/ohe2.png')
    st.image(image1, use_column_width=True)

    df = pd.read_csv('data/breast-cancer.csv')
    
    df.rename(columns={'Class': 'target'}, inplace=True)

    df.fillna('unknown', inplace = True)
    
    feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))
    
    st.dataframe(df[f'{feat}'])

    button2 = st.button('Apply One Hot Encoding')
    if button2:
        X_ohe = pd.get_dummies(df[feat])
        X_ohe_all = pd.DataFrame(df[feat])
        X_ohe_all = pd.concat([X_ohe_all,X_ohe], axis=1)
        st.dataframe(X_ohe_all)

    showImplementation = st.checkbox('Show Code', key='ShowCodeOneHot') 
    from sklearn.preprocessing import OneHotEncoder
    
    if showImplementation:
        st.subheader("`sklearn` has a special module for One Hot Encoder:")
        with st.echo():
            from sklearn.preprocessing import OneHotEncoder
            ohe = OneHotEncoder()
            X_ohe = ohe.fit_transform(df[[feat]])

        st.subheader("Addition to `sklearn`, `pandas` has also a module `get_dummies` for One Hot Encoding")
        
        with st.echo():
            X_ohe = pd.get_dummies(df[feat])
            X_ohe_all = pd.DataFrame(df[feat])
            X_ohe_all = pd.concat([X_ohe_all,X_ohe], axis=1)
    
    
    st.warning(""" 
    ### :exclamation: Think About
    - How similar/dissimilary each value in breast-quad with respect to each other ?
    - Note that the answer of question is mainly related with the encoding you use.
    """)

    st.markdown('### 1. Similarity/Dissimilarity each value in `breast-quad` wrt each other by LabelEncoder')

    st.markdown('Label Encoder applied to `breast-quad`:')

    df['breast-quad'].unique()
        
    #st.markdown('Unique values are')
    #st.write(df['breast-quad'].unique())
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    enc_le = le.fit(df['breast-quad']).transform(df['breast-quad'].unique()).reshape(-1,1)
    
   
    #button = st.button('Apply Label Encoding for Unique Values')
    
    #if button:
    #st.subheader('Mapped Values')
    st.write( enc_le)
            
            
    st.info(":pushpin:  Pairwise methods evaluate all pairs of sequences and transform the differences into a distance.")

    st.markdown('Pairwise distances of the result:')
    from sklearn.metrics import pairwise_distances
    st.write(pairwise_distances(enc_le.reshape(-1,1)))
    
    if showImplementation:
        st.markdown('#### Calculate Pairwise Distance Between Label Encoding of Each Unique Value')
        with st.echo():
            pairwise_distances(enc_le.reshape(-1,1))
        st.write(pairwise_distances(enc_le.reshape(-1,1)))

    buttonShowCodePairwise1 = st.button('Show Code', key="pairwiseLabel")
    if buttonShowCodePairwise1:
             with st.echo():
                from sklearn.preprocessing import LabelEncoder
                # apply label encoding
                le = LabelEncoder()
                enc_le = le.fit(df['breast-quad']).transform(df['breast-quad'].unique()).reshape(-1,1)

                # calculate pairwise distances
                from sklearn.metrics import pairwise_distances
                pairwise_distances(enc_le.reshape(-1,1))


    st.markdown('### 2. Similarity/Dissimilarity each value in `breast-quad` wrt each other by One Hot Encoder')
                                   
    ohe = OneHotEncoder()
    enc_ohe = ohe.fit(df[['breast-quad']]).transform(df['breast-quad'].unique().reshape(-1,1)).toarray()
    
    st.markdown('One Hot Encoding applied to `breast-quad`:')
    st.write(enc_ohe)
    st.markdown('Pairwise distances of the result:')
    st.write(pairwise_distances(enc_ohe))

    buttonShowCodePairwise2 = st.button('Show Code', key="pairwiseOhe")
    if buttonShowCodePairwise2:
             with st.echo():
                # apply one hot encoder
                ohe = OneHotEncoder()
                enc_ohe = ohe.fit(df[['breast-quad']]).transform(df['breast-quad'].unique().reshape(-1,1)).toarray()

                #calculate pairwise distances
                from sklearn.metrics import pairwise_distances
                pairwise_distances(enc_ohe)


    st.success('Hence our new encoding preserves relative similarity/dissimilarity of each unique value :white_check_mark:')