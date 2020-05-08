import streamlit as st
import pandas as pd
import numpy as np 
from PIL import Image

def content():

    df = pd.read_csv('data/breast-cancer.csv')
    
    df.rename(columns={'Class': 'target'}, inplace=True)
    df.fillna('unknown', inplace = True)
    
    st.title("Weight of Evidence Encoding")

    st.markdown(" * Weight of Evidence (WoE) is a measure of the “strength” of a grouping technique to separate good and bad.")

    image = Image.open('images/woe3.jpg')
    st.image(image)

    st.markdown(" * Weight of evidence (WOE) is a measure of how much the evidence supports or undermines a hypothesis.")

    image2 = Image.open('images/woe.jpg')
    st.image(image2, use_column_width=True)

    st.info(":gem: This method was developed primarily to build a predictive model to evaluate the risk of loan default in the credit and financial industry.")

    st.warning(":pushpin:  WoE is well suited for Logistic Regression because the Logit transformation is simply the log of the odds, i.e., ln(P(Goods)/P(Bads)).")


    feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

    st.dataframe(df[feat])
    
    st.warning(":exclamation: First we need to convert target categories to numeric values.")
    
    st.subheader("Simply replace")
    with st.echo():
        df['target'].replace({'recurrence-events': 1, 'no-recurrence-events': 0}, inplace=True)
    
    st.subheader("`mlencoders` has a module for Weight Of Evidence Encoding")
    
    with st.echo():
        from mlencoders.weight_of_evidence_encoder import WeightOfEvidenceEncoder
        weightEnc = WeightOfEvidenceEncoder()
        
    # For corner cases, defaulting to WOE = 0 (meaning no info). To avoid division by 0 we use default values.
    #undef = (mapping['count'] < self.min_samples) | (mapping['pos'] == 0) | (mapping['neg'] == 0)
    #mapping.loc[undef, ['pos', 'neg']] = -1
    # Final step, log of ratio of probabily estimates
    
    showImplementation = st.checkbox('Steps of Weight Of Evidence Encoding', key='key2') 
    
    if showImplementation:
        st.subheader(" **Step1:** Calculate Events and All Events")
        with st.echo():
            mapping = df['target'].groupby(df[feat]).agg(['sum', 'count']).rename({'sum': 'events'}, axis=1)
        st.write(mapping)
        st.subheader(" **Step2:** Calculate Non-Events")
        with st.echo():
            mapping['non_events'] = mapping['count'] - mapping['events']
        st.write(mapping)
        st.subheader(" **Step3:** Calculate % of Events and % of Non-Events")
        with st.echo():
            mapping[['%_of_events','%_of_non_events']] = mapping[['events', 'non_events']]/mapping[['events', 'non_events']].sum()
        st.write(mapping)
        st.subheader(" **Step4:** Calculate Weight Of Evidence Values")
        with st.echo():
            mapping['WoE'] = np.log(mapping['%_of_events'] / mapping['%_of_non_events'])
        st.write(mapping)
    
        
    X_woe = pd.DataFrame(df[feat])
    X_woe[f'tranformed_{feat}'] = weightEnc.fit_transform(df[[feat]],df['target'])

    showImplementation = st.checkbox('Show Code', key='key1') 
    
    if showImplementation:
        with st.echo():
             X_woe[f'tranformed_{feat}'] = weightEnc.fit_transform(df[[feat]],df['target'])
                
    button = st.button('Apply Weight of Evidence Encoding')
    if button:
        st.dataframe(X_woe)