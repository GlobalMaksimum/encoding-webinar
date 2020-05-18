import streamlit as st
import pandas as pd
import numpy as np 
from PIL import Image

def content():

    df = pd.read_csv('data/breast-cancer.csv')
    
    df.rename(columns={'Class': 'target'}, inplace=True)
    df.fillna('unknown', inplace = True)
    df['deg-malig'] = df['deg-malig'].astype('object')

    st.title("Binary Encoding")

    st.markdown(" * Binary encoding converts a category into binary digits. Each binary digit creates one feature column.")

    image3 = Image.open('images/ohe_binary.png')
    st.image(image3, use_column_width=True)

    st.warning(":exclamation:  Compared to one hot encoding, this will require fewer feature columns.")


    feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

    st.dataframe(df[feat])

    X_binary_all = pd.DataFrame(df[feat])
    import category_encoders as ce
    binaryEnc = ce.BinaryEncoder()
    
    button = st.button('Apply Binary Encoding')
    if button:
        X_binary = binaryEnc.fit_transform(df[feat])
        X_binary_all = pd.concat([X_binary_all,X_binary], axis=1)
        st.dataframe(X_binary_all)

    showImplementation = st.checkbox('Show Code', key='key1') 
    
    if showImplementation:
        st.subheader("`category_encoders` has a module for Binary Encoding")
        with st.echo():
            
            import category_encoders as ce
            binaryEnc = ce.BinaryEncoder()
            X_binary = binaryEnc.fit_transform(df[feat])

    
    
   