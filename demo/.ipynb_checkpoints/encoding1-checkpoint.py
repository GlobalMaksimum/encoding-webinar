import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.title("A Quick Look at Encoding")

st.markdown("""
         Encoding is the process of converting the data or a given sequence of characters, symbols, alphabets etc., into a specified format, for the secured transmission of data.
        """)

st.header('We Usually Have Categorical Variables in Our Dataset')

st.markdown('A categorical variable is a variable that can take on one of a limited, and usually fixed number of possible values.')

st.header("Categorical Variables can be divided into 2 categories:")

image1 = Image.open('../images/cat.png')

st.image(image1, use_column_width=True)

st.markdown('Nominal variable that has no numerical importance, besides ordinal variable has some order.')

st.header('We Need to Apply Appropriate Encoding for Categorical Variables. Why?')

st.markdown("""
         Most of the Machine Learning algorithms can not handle Categorical Variables unless we convert them to numerical values.
        """)

st.title('Now We are Applying Encodings What We Learned so Far')

df = pd.read_csv('../data/breast-cancer.csv')

st.markdown("Let's define target for our classification problem.")

with st.echo():
    df.rename(columns={'Class': 'target'}, inplace=True)

df.fillna('unknown', inplace = True)

st.header('Select Binary Columns')

bin_cols = st.multiselect('Binary Cols: ',df.columns.tolist())

st.dataframe(df[bin_cols].head(10))

st.header('Simple Idea: Replacing Values')

st.markdown('We can simply replace categorical variables with specific numerical values.')

with st.echo():
    df['breast'].replace({'right': 1, 'left': 0}, inplace=True)
    df['irradiat'].replace({'yes': 1, 'no': 0}, inplace=True)
    df['target'].replace({'recurrence-events': 1, 'no-recurrence-events': 0}, inplace=True)
        
button1 = st.button('Replace Values')
if button1:
    st.dataframe(df[bin_cols])

st.header('Select Nominal Columns')

nom_cols = st.multiselect('Nominal Cols: ',df.columns.tolist())

st.dataframe(df[nom_cols].head(10))

st.header('Select Ordinal Columns')

ord_cols = st.multiselect('Ordinal Cols: ',df.columns.tolist())

st.dataframe(df[ord_cols].head(10))

st.header("Let's Apply Label Encoding for Nominal Columns")

with st.echo():
    from sklearn.preprocessing import LabelEncoder

with st.echo():
    X_label = df[nom_cols].copy()
    
st.dataframe(X_label.head(10))

with st.echo():
    for col in nom_cols:
        labelEnc = LabelEncoder()
        X_label[col] = labelEnc.fit_transform(X_label[col])

button2 = st.button('Apply Label Encoding')
if button2:
    st.dataframe(X_label)

st.header("Let's Apply Ordinal Encoding for Ordinal Columns")

with st.echo():
    from sklearn.preprocessing import OrdinalEncoder

with st.echo():
    X_ord = df[ord_cols].copy()
    
st.dataframe(X_ord.head(10))

st.markdown("We need to give order between categories")

with st.echo():
    ordEnc = OrdinalEncoder(categories=[['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']])
    X_ord[['age']] = ordEnc.fit_transform(X_ord[['age']])
                                    
    ordEnc = OrdinalEncoder(categories=[['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54']])
    X_ord[['tumor-size']] = ordEnc.fit_transform(X_ord[['tumor-size']])

    ordEnc = OrdinalEncoder(categories=[['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '24-26']])
    X_ord[['inv-nodes']] = ordEnc.fit_transform(X_ord[['inv-nodes']])

    ordEnc = OrdinalEncoder()
    X_ord[['deg-malig']] = ordEnc.fit_transform(X_ord[['deg-malig']])

button3 = st.button('Apply Ordinal Encoding')
if button3:
    st.dataframe(X_ord)
    
X_label.to_csv('X_label.csv', index=False)
X_ord.to_csv('X_ord.csv', index=False)