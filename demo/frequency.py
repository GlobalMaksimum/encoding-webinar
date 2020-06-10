import streamlit as st
import pandas as pd
import numpy as np 
from PIL import Image

def content():

    df = pd.read_csv('data/breast-cancer.csv')
    
    df.rename(columns={'Class': 'target'}, inplace=True)
    df.fillna('unknown', inplace = True)
    
    st.title("Frequency Encoding")

    st.markdown(" * Frequency encoding is a way to utilize frequencies of categories as labels.")
    st.markdown(" * It can help if frequency correlates with the target and also, it can help the model to understand that smaller categories are less trustworthy then bigger ones, especially when frequency encoding is used parallel with other type of encoding.")

    image1 = Image.open('images/freq.jpeg')
    st.image(image1, use_column_width=True)

    st.info(":pushpin:  In the cases where the frequency is related somewhat with the target variable, it helps the model to understand and assign the weight in direct and inverse proportion, depending on the nature of the data.")

    feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

    X_freq = pd.DataFrame(df[feat])

    st.dataframe(X_freq)

    freqEnc = (X_freq.groupby(feat).size()) / len(X_freq)
    freqEnc2 = pd.DataFrame(freqEnc)
    freqEnc2.columns = ['Frequency']

    st.dataframe(pd.DataFrame(freqEnc2))

    X_freq[f'tranformed_{feat}'] = X_freq[feat].apply(lambda x : freqEnc[x])

    
    button = st.button('Apply Frequency Encoding')
    if button:
        st.dataframe(X_freq)

    showImplementation = st.checkbox('Show Code', key='key1') 
    
    if showImplementation:
        X_freq = pd.DataFrame(df[feat])
        with st.echo():
            freqEnc = (X_freq.groupby(feat).size()) / len(X_freq)
            X_freq[f'tranformed_{feat}'] = X_freq[feat].apply(lambda x : freqEnc[x])
            
    st.markdown(" * Faster learning than with one-hot representations. Numbers in  R^1  are way more compact representations then vectors in R^n−1 used in one-hot encoding, that leads to fewer features for trees, which leads to faster learning.")
    st.markdown(" * Less splits needed, means a more robust model. Unlike with one hot encoding, here trees can separate several categories at a time (with one-hot it is always 1).")
    st.markdown(" * Frequency encoding is a way to utilize frequencies of categories as labels.")
