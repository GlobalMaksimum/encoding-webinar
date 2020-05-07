import streamlit as st
import pandas as pd
import numpy as np 
from PIL import Image

def content():
    
    df = pd.read_csv('data/train.csv')

    st.title("Cyclic Encoding")

    st.markdown("""
            The transformation of cyclic features is important because when cyclic features are untransformed then there's no way for the model to understand that the smallest value in the cycle is actually next to the largest value.
            """)

    st.header('What is Cyclic Encoding?')

    st.markdown("""
             The main idea behind cyclic encoding is to enable cyclic data to be represented on a circle.
            """)

    image = Image.open('images/cyclic2.png')

    st.image(image, use_column_width=True)

    st.markdown("""
             Examples are 'day' and 'month' features.
            """)

    st.dataframe(df[['day','month']])

    st.header("Let's Map These Features onto The Unit Circle As in Below.")

    with st.echo():
        X_cyclic['day_sin'] = np.sin(2 * np.pi * df['day']/7)
        X_cyclic['day_cos'] = np.cos(2 * np.pi * df['day']/7)
        X_cyclic['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        X_cyclic['month_cos'] = np.cos(2 * np.pi * df['month']/12)

    #X_cyclic.drop(['day','month'], axis=1, inplace=True)

    button = st.button('Apply Cyclic Encoding')
    if button:
        st.dataframe(X_cyclic)