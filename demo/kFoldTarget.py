import streamlit as st
import pandas as pd
import numpy as np 
from PIL import Image

def content():

    df = pd.read_csv('data/breast-cancer.csv')
    
    df.rename(columns={'Class': 'target'}, inplace=True)
    df.fillna('unknown', inplace = True)
    
    st.header("K-Fold Target Encoding")

    st.markdown("By dividing the data into folds we can reduce the overfitting.")

    image = Image.open('images/kfold.png')
    st.image(image, use_column_width=True)

    from sklearn.model_selection import KFold

    feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

    st.dataframe(df[feat])

    X_kfold = pd.DataFrame(df[[feat,'target']])

    kf = KFold(n_splits = 5, shuffle = False)
    fold_df = pd.DataFrame()

    for train_ind,val_ind in kf.split(X_kfold):
        if(X_kfold[feat].dtype == 'object') and 'tenc' not in feat:
            replaced = X_kfold.iloc[train_ind][[feat,'target']].groupby(feat)['target'].mean()
            fold_df = pd.concat([fold_df, pd.DataFrame(replaced)], axis = 1)
            replaced = dict(replaced)
            X_kfold.loc[val_ind,f'tenc_{feat}'] = X_kfold.iloc[val_ind][feat].replace(replaced).values

    fold_df.columns = ['fold_1','fold_2','fold_3','fold_4','fold_5']

    button = st.button('Mean of Targets in K-Fold')
    if button:
        st.dataframe(fold_df)

    X_kfold.drop([feat,'target'], axis=1, inplace=True)

    button2 = st.button('Apply K-Fold Target Encoding')
    if button2:
        st.dataframe(X_kfold)