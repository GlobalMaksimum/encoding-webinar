import streamlit as st
import pandas as pd
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt

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
    
    targetSum = df.groupby(feat)['target'].agg('sum')
    targetCount = df.groupby(feat)['target'].agg('count')
    targetEnc2 = targetSum/targetCount

    steps = pd.DataFrame(targetSum)
    steps = pd.concat([steps,targetCount], axis=1)
    steps = pd.concat([steps,targetEnc2], axis=1)
    steps.columns = ['Sum of Targets','Count of Targets','Mean of Targets']

    st.subheader("Steps for calculating Mean of Targets")
    st.dataframe(steps)
    
    targetEnc2 = dict(targetEnc2)
    X_target = pd.DataFrame(df[feat])
    X_target[f'tranformed_{feat}'] = df[feat].replace(targetEnc2).values
    
    button = st.button('Apply Target Encoding')
    if button:
        st.dataframe(X_target)

    st.warning(":exclamation: Because we don't know targets of the test data this can lead the overfitting.")
    showImplementation = st.checkbox('Show Code', key='key1') 
    
    if showImplementation:
        st.subheader("`category_encoders` has a module for Target Encoding")
        with st.echo():
            import category_encoders as ce
            targetEnc = ce.TargetEncoder()
            X_target[f'tranformed_{feat}'] = targetEnc.fit_transform(df[feat],df['target'])
        
        
    df_all = pd.DataFrame(df[feat])
    df_all.columns = ['feature']
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_all[f'feature_label'] = le.fit_transform(df_all['feature'])
    df_all[f'feature_mean'] = df_all['feature'].replace(targetEnc2).values
    df_all['target'] = df['target']
    st.dataframe(df_all)
    
    #df_all.groupby("target").feature_label.hist(alpha=0.4)
    #st.pyplot()

    #df_all.groupby("target").feature_mean.hist(alpha=0.4)
    #st.pyplot()
    
    st.markdown(""" * Label encoding gives random order. No correlation with target.""")
    st.markdown(""" * Mean encoding helps to separate zeros from ones.""")

    def sephist(col):
        target_1 = df_all[df_all['target'] == 1][col]
        target_0 = df_all[df_all['target'] == 0][col]
        return target_1, target_0

    for num, alpha in enumerate(['feature_label','feature_mean']):
        plt.subplot(1, 2, num+1)
        plt.hist((sephist(alpha)[0], sephist(alpha)[1]), alpha=0.5, label=['target_1', 'target_0'], color=['r', 'b'])
        #plt.hist(sephist(alpha)[0], alpha=0.5, label='target_1', color='b')
        #plt.hist(sephist(alpha)[1], alpha=0.5, label='target_0', color='r')
        plt.legend(loc='upper right')
        plt.title(alpha)
    plt.tight_layout(pad=1)
    st.pyplot()

    image2 = Image.open('images/target1.png')
    st.image(image2, use_column_width=True)
    
    image3 = Image.open('images/target2.png')
    st.image(image3, use_column_width=True)
    
    image4 = Image.open('images/target3.png')
    st.image(image4, use_column_width=True)
    
    st.warning(":exclamation: Because we don't know targets of the test data this can lead the overfitting.")

