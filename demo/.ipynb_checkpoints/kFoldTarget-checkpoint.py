import streamlit as st
import pandas as pd
import numpy as np 
from PIL import Image

def content():

    df = pd.read_csv('data/breast-cancer.csv')
    
    df.rename(columns={'Class': 'target'}, inplace=True)
    df.fillna('unknown', inplace = True)
    
    st.title("K-Fold Target Encoding")
    
    st.markdown(""" 
    
    
    """)

    st.info(":pushpin: By dividing the data into folds we can reduce the overfitting.")

    image = Image.open('images/kfold.png')
    st.image(image, use_column_width=True)

    from sklearn.model_selection import KFold

    feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

    st.dataframe(df[feat])

    st.warning(":exclamation: First we need to convert target categories to numeric values.")
    
    st.subheader("Simply replace")
    with st.echo():
        df['target'].replace({'recurrence-events': 1, 'no-recurrence-events': 0}, inplace=True)
        
    #X_kfold = pd.DataFrame(df[[feat,'target']])
    X_kfold = pd.DataFrame(df[feat])
    
    kf = KFold(n_splits = 5, shuffle = False)
    fold_df = pd.DataFrame()

    for train_ind,val_ind in kf.split(df):
        if 'transformed' not in feat:
            replaced = df.iloc[train_ind][[feat,'target']].groupby(feat)['target'].mean()
            fold_df = pd.concat([fold_df, pd.DataFrame(replaced)], axis = 1)
            replaced = dict(replaced)
            X_kfold.loc[val_ind,f'tranformed_{feat}'] = df.iloc[val_ind][feat].replace(replaced).values

    #X_kfold.loc[val_ind,f'tenc_{feat}'] = df[feat].iloc[val_ind][feat].replace(replaced).values
    fold_df.columns = ['fold_1','fold_2','fold_3','fold_4','fold_5']

    st.subheader(':small_orange_diamond: Mean of Targets in K-Fold')
    st.dataframe(fold_df)

    #X_kfold.drop([feat], axis=1, inplace=True)
    
    
    showImplementation = st.checkbox('Show Code', key='key1') 
    
    if showImplementation:
        X_kfold = pd.DataFrame(df[feat])
        with st.echo():
            kf = KFold(n_splits = 5, shuffle = False)

            for train_ind,val_ind in kf.split(df):
                if 'transformed' not in feat:
                    replaced = dict(df.iloc[train_ind][[feat,'target']].groupby(feat)['target'].mean())
                    X_kfold.loc[val_ind,f'transformed_{feat}'] = df.iloc[val_ind][feat].replace(replaced).values
        #X_kfold.drop([feat], axis=1, inplace=True)

    button = st.button('Apply K-Fold Target Encoding')
    if button:
        st.dataframe(X_kfold)