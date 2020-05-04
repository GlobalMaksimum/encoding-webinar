import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

df = pd.read_csv('../data/train.csv')

X_cyclic = df[['day','month']].copy()

st.title("We Can Apply Speacial Encoding for Cyclic Features")

st.markdown("""
        The transformation of cyclic features is important because when cyclic features are untransformed then there's no way for the model to understand that the smallest value in the cycle is actually next to the largest value.
        """)

st.header('What is Cyclic Encoding?')

st.markdown("""
         The main idea behind cyclic encoding is to enable cyclic data to be represented on a circle.
        """)

st.header("For Example 'hours' Variable is a Cyclic Feature.")

st.markdown("""
         We map each cyclical variable onto a circle such that the lowest value for that variable appears right next to the largest value. We compute the x- and y- component of that point using sin and cos trigonometric functions. You remember your unit circle, right? Here's what it looks like for the 'hours' variable. Zero (midnight) is on the right, and the hours increase counterclockwise around the circle. In this way, 23:59 is very close to 00:00, as it should be.
    """)

image = Image.open('../images/cyclic2.png')

st.image(image)

st.markdown("""
         Another examples are 'day' and 'month' features.
        """)

st.dataframe(X_cyclic)

st.header("Let's Map These Features onto The Unit Circle As in Below.")

with st.echo():
    X_cyclic['day_sin'] = np.sin(2 * np.pi * X_cyclic['day']/7)
    X_cyclic['day_cos'] = np.cos(2 * np.pi * X_cyclic['day']/7)
    X_cyclic['month_sin'] = np.sin(2 * np.pi * X_cyclic['month']/12)
    X_cyclic['month_cos'] = np.cos(2 * np.pi * X_cyclic['month']/12)

X_cyclic.drop(['day','month'], axis=1, inplace=True)

button = st.button('Apply Cyclic Encoding')
if button:
    st.dataframe(X_cyclic)