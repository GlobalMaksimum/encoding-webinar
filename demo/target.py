import streamlit as st
import pandas as pd
import numpy as np 
from PIL import Image

def content():

    df = pd.read_csv('data/breast-cancer.csv')
    
    df.rename(columns={'Class': 'target'}, inplace=True)
    df.fillna('unknown', inplace = True)
    
    st.header("Target (Mean) Encoding")

    st.markdown("In Target Encoding for each category in the feature label is decided with the mean value of the target variable on a training data.")

    image = Image.open('images/target.png')
    st.image(image, use_column_width=True)

    st.markdown("This encoding method brings out the relation between similar categories.")

    feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

    st.dataframe(df[feat])

    X_target = pd.DataFrame(df[[feat,'target']])

    targetSum = X_target.groupby(feat)['target'].agg('sum')
    targetCount = X_target.groupby(feat)['target'].agg('count')
    targetEnc = targetSum/targetCount

    steps = pd.DataFrame(targetSum)
    steps = pd.concat([steps,targetCount], axis=1)
    steps = pd.concat([steps,targetEnc], axis=1)
    steps.columns = ['Sum of Targets','Count of Targets','Mean of Targets']

    button = st.button('Steps for Target Encoding')
    if button:
        st.dataframe(steps)

    targetEnc = dict(targetEnc)
    X_target[feat] = X_target[feat].replace(targetEnc).values

    X_target.drop(['target'], axis=1, inplace=True)

    button2 = st.button('Apply Target Encoding')
    if button2:
        st.dataframe(X_target)

    st.markdown("Because we don't know targets of the test data this can lead the overfitting.")