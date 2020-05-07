import streamlit as st
import pandas as pd
import numpy as np 
from PIL import Image

def content():

    df = pd.read_csv('data/breast-cancer.csv')
    
    df.rename(columns={'Class': 'target'}, inplace=True)
    df.fillna('unknown', inplace = True)
    
    st.header("Weight of Evidence Encoding")

    st.markdown("Weight of Evidence (WoE) is a measure of the “strength” of a grouping technique to separate good and bad.")

    image = Image.open('images/woe3.jpg')
    st.image(image, use_column_width=True)

    st.markdown("Weight of evidence (WOE) is a measure of how much the evidence supports or undermines a hypothesis.")

    image2 = Image.open('images/woe.jpg')
    st.image(image2, use_column_width=True)

    st.markdown("This method was developed primarily to build a predictive model to evaluate the risk of loan default in the credit and financial industry.")

    st.markdown("WoE is well suited for Logistic Regression because the Logit transformation is simply the log of the odds, i.e., ln(P(Goods)/P(Bads)).")

    from mlencoders.weight_of_evidence_encoder import WeightOfEvidenceEncoder

    feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

    st.dataframe(df[feat])

    weightEnc = WeightOfEvidenceEncoder()
    X_woe = weightEnc.fit_transform(df[feat],df['target'])

    button = st.button('Apply Weight of Evidence Encoding')
    if button:
        st.dataframe(X_woe)

    #X_woe_ord[['deg-malig']] = X_woe_ord[['deg-malig']].astype('object')