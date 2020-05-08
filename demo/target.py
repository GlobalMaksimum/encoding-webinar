import streamlit as st
import pandas as pd
import numpy as np 
from PIL import Image

def content():

    df = pd.read_csv('data/breast-cancer.csv')
    
    df.rename(columns={'Class': 'target'}, inplace=True)
    df.fillna('unknown', inplace = True)
    
    st.title("Target (Mean) Encoding")

    st.markdown(" * In Target Encoding for each category in the feature label is decided with the mean value of the target variable on a training data.")

    image = Image.open('images/target.png')
    st.image(image, use_column_width=True)

    st.info(":pushpin:  This encoding method brings out the relation between similar categories.")

    feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

    st.dataframe(df[feat])
    
    st.warning(":exclamation: First we need to convert target categories to numeric values.")
    
    st.subheader("Simply replace")
    with st.echo():
        df['target'].replace({'recurrence-events': 1, 'no-recurrence-events': 0}, inplace=True)
        
    st.subheader("`category_encoders` has a module for Target Encoding")
    
    with st.echo():
        import category_encoders as ce
        targetEnc = ce.TargetEncoder()
    
    targetSum = df.groupby(feat)['target'].agg('sum')
    targetCount = df.groupby(feat)['target'].agg('count')
    targetEnc2 = targetSum/targetCount

    steps = pd.DataFrame(targetSum)
    steps = pd.concat([steps,targetCount], axis=1)
    steps = pd.concat([steps,targetEnc2], axis=1)
    steps.columns = ['Sum of Targets','Count of Targets','Mean of Targets']

    st.subheader(":small_orange_diamond: Steps for calculating Mean of Targets")
    st.dataframe(steps)

    targetEnc2 = dict(targetEnc2)
    X_target = pd.DataFrame(df[feat])
    X_target[f'tranformed_{feat}'] = df[feat].replace(targetEnc2).values
    
    showImplementation = st.checkbox('Show Code', key='key1') 
    
    if showImplementation:
        with st.echo():
             X_target[f'tranformed_{feat}'] = targetEnc.fit_transform(df[feat],df['target'])
            
    button = st.button('Apply Target Encoding')
    if button:
        st.dataframe(X_target)

    st.warning(":exclamation: Because we don't know targets of the test data this can lead the overfitting.")