import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

def content():


    df = pd.read_csv('data/breast-cancer.csv')

    st.header('How to Generalize Dictionary Encoding into N Distinct Values?')


    image1 = Image.open('images/label.png')
    st.image(image1, use_column_width=True)

    feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

    st.dataframe(df[f'{feat}'])

    from sklearn.preprocessing import LabelEncoder  
    le = LabelEncoder()
    X_label = pd.DataFrame(df[feat])
    X_label[f'tranformed_{feat}'] = le.fit_transform(df[feat])


    button2 = st.button('Apply Label Encoding')
    if button2:
        st.dataframe(X_label)

    showImplementation = st.checkbox('Show Code', key='showCodeLabelEnc') 
    
    if showImplementation:
        with st.echo():
            df[feat].unique()
            #fill missing values
            df[feat] = df[feat].fillna('unknown')
    
        st.markdown('Unique values are')
        st.write(df[feat].unique())

    
        st.subheader('`sklearn` has a great module for this: Label Encoder')

        with st.echo():
            from sklearn.preprocessing import LabelEncoder  
  
            X_label = pd.DataFrame(df[feat])

            le = LabelEncoder()
            X_label[f'tranformed_{feat}'] = le.fit_transform(df[feat])
    
    
    
   
        
    st.warning("""
     :question: What are the potential issues with Label Encoder approach?
    """)