import streamlit as st
import pandas as pd
import numpy as np 
from PIL import Image

def content():

    df = pd.read_csv('data/breast-cancer.csv')
    
    df.rename(columns={'Class': 'target'}, inplace=True)
    df.fillna('unknown', inplace = True)

    #st.header("We Can Reduce Dimensionality Using Binary Encoding")

    st.header("Binary Encoding")

    st.markdown("Binary encoding converts a category into binary digits. Each binary digit creates one feature column.")

    image3 = Image.open('images/ohe_binary.png')
    st.image(image3, use_column_width=True)

    st.markdown("Compared to one hot encoding, this will require fewer feature columns.")

    import category_encoders as ce

    #st.markdown("Here we apply binary encoding for ordinal columns.")

    feat2 = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

    X_binary = pd.DataFrame(df[feat2])

    st.dataframe(X_binary)

    #X_binary[['deg-malig']] = X_binary[['deg-malig']].astype('object')

    binaryEnc = ce.BinaryEncoder()
    X_binary = binaryEnc.fit_transform(X_binary)

    button3 = st.button('Apply Binary Encoding')
    if button3:
        st.dataframe(X_binary)