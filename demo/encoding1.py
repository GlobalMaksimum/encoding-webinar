import streamlit as st
import pandas as pd
import numpy as np

st.title('Now We are Applying What We Learned so Far')

df = pd.read_csv('../data/breast-cancer.csv')

with st.echo():
    df.rename(columns={'Class': 'target'}, inplace=True)
    
st.header('Select Binary Cols')

bin_feats = st.multiselect('Binary Cols: ',df.columns.tolist())

df_bin = df[bin_feats]

st.dataframe(df[bin_feats].head(10))



with st.echo():
    df['breast'].replace({'right': 1, 'left': 0}, inplace=True)
    df['irradiat'].replace({'yes': 1, 'no': 0}, inplace=True)
    df['target'].replace({'recurrence-events': 1, 'no-recurrence-events': 0}, inplace=True)
    
    
button1 = st.button('Transform')
if button1:
    st.dataframe(df[bin_feats])


st.header('Select Nominal Cols')

nom_feats = st.multiselect('Nominal Cols: ',df.columns.tolist())

df_nom = df[nom_feats]

st.dataframe(df[nom_feats].head(10))