import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

df = pd.read_csv('../data/breast-cancer.csv')

df.rename(columns={'Class': 'target'}, inplace=True)
df.fillna('unknown', inplace = True)
bin_cols = ['breast','irradiat']
nom_cols = ['breast-quad','menopause','node-caps']
ord_cols = ['age','tumor-size','inv-nodes','deg-malig']
df['breast'].replace({'right': 1, 'left': 0}, inplace=True)
df['irradiat'].replace({'yes': 1, 'no': 0}, inplace=True)
df['target'].replace({'recurrence-events': 1, 'no-recurrence-events': 0}, inplace=True)

st.header("Frequency Encoding is Another Encoding Technique That Utilize The Frequency of The Categories as Labels.")

#st.markdown("It is a way to utilize the frequency of the categories as labels.")

image1 = Image.open('../images/freq.jpeg')
st.image(image1, use_column_width=True)

st.markdown("In the cases where the frequency is related somewhat with the target variable, it helps the model to understand and assign the weight in direct and inverse proportion, depending on the nature of the data.")

st.markdown("If we apply frequency encoding for nonminal columns:")

with st.echo():
    X_freq = df[nom_cols].copy()
    
st.dataframe(X_freq.head(10))

with st.echo():
    for col in X_freq.columns:
        freqEnc = (X_freq.groupby(col).size()) / len(X_freq)
        X_freq[col] = X_freq[col].apply(lambda x : freqEnc[x])
    
button1 = st.button('Apply Frequency Encoding')
if button1:
    st.dataframe(X_freq)

st.header("We Can Replace A Categorical Value with The Mean of The Target Variable Using Target (Mean) Encoding")

st.markdown("Target Encoding is similar to Label Encoding, except here labels are correlated directly with the target.")

st.markdown("In Target Encoding for each category in the feature label is decided with the mean value of the target variable on a training data.")

image2 = Image.open('../images/target.png')
st.image(image2, use_column_width=True)

st.markdown("This encoding method brings out the relation between similar categories.")

with st.echo():
    X_target_nom = df[nom_cols].copy()
    
st.dataframe(X_target_nom.head(10))

X_target_nom = df[nom_cols+['target']].copy()

with st.echo():
    for col in nom_cols:
        targetEnc = dict(X_target_nom.groupby(col)['target'].agg('sum')/X_target_nom.groupby(col)['target'].agg('count'))
        X_target_nom[col] = X_target_nom[col].replace(targetEnc).values

X_target_nom.drop(['target'], axis=1, inplace=True)
        
button2 = st.button('Apply Target Encoding')
if button2:
    st.dataframe(X_target_nom)
    
X_target_ord = df[ord_cols+['target']].copy()
X_target_ord[['deg-malig']] = X_target_ord[['deg-malig']].astype('object')
for col in ord_cols:
    targetEnc = dict(X_target_ord.groupby(col)['target'].agg('sum')/X_target_ord.groupby(col)['target'].agg('count'))
    X_target_ord[col] = X_target_ord[col].replace(targetEnc).values
X_target_ord.drop(['target'], axis=1, inplace=True)

st.markdown("Because we don't know targets of the test data this can lead the overfitting.")

st.header("The Solution to Prevent Overfitting: K-Fold Target Encoding")

st.markdown("By dividing the data into folds we can reduce the overfitting.")

image3 = Image.open('../images/kfold.png')
st.image(image3, use_column_width=True)

from sklearn.model_selection import KFold
    
with st.echo():
    X_kfold_nom = df[nom_cols].copy()
    
st.dataframe(X_kfold_nom.head(10))

X_kfold_nom = df[nom_cols+['target']].copy()

with st.echo():
    kf = KFold(n_splits = 5, shuffle = False)

    for train_ind,val_ind in kf.split(X_kfold_nom):
        for col in nom_cols:
            if(X_kfold_nom[col].dtype == 'object') and 'tenc' not in col:
                replaced = dict(X_kfold_nom.iloc[train_ind][[col,'target']].groupby(col)['target'].mean())
                X_kfold_nom.loc[val_ind,f'tenc_{col}'] = X_kfold_nom.iloc[val_ind][col].replace(replaced).values

X_kfold_nom.drop(nom_cols+['target'], axis=1, inplace=True)
        
button4 = st.button('Apply K-Fold Target Encoding')
if button4:
    st.dataframe(X_kfold_nom)

X_kfold_ord = df[ord_cols+['target']].copy()

kf = KFold(n_splits = 5, shuffle = False)

for train_ind,val_ind in kf.split(X_kfold_ord):
    for col in ord_cols:
        if(X_kfold_ord[col].dtype == 'object') and 'tenc' not in col:
            replaced = dict(X_kfold_ord.iloc[train_ind][[col,'target']].groupby(col)['target'].mean())
            X_kfold_ord.loc[val_ind,f'tenc_{col}'] = X_kfold_ord.iloc[val_ind][col].replace(replaced).values

X_kfold_ord.drop(ord_cols+['target'], axis=1, inplace=True)
            
st.header("Another Encoding Technique is Weight of Evidence Encoding")

st.markdown("Weight of Evidence (WoE) is a measure of the “strength” of a grouping technique to separate good and bad.")

image4 = Image.open('../images/woe3.jpg')
st.image(image4, use_column_width=True)

st.markdown("Weight of evidence (WOE) is a measure of how much the evidence supports or undermines a hypothesis.")

image5 = Image.open('../images/woe.jpg')
st.image(image5, use_column_width=True)

st.markdown("This method was developed primarily to build a predictive model to evaluate the risk of loan default in the credit and financial industry.")

st.markdown("WoE is well suited for Logistic Regression because the Logit transformation is simply the log of the odds, i.e., ln(P(Goods)/P(Bads)).")

with st.echo():
    from mlencoders.weight_of_evidence_encoder import WeightOfEvidenceEncoder
    
with st.echo():
    X_woe_nom = df[nom_cols].copy()
    
st.dataframe(X_woe_nom.head(10))

with st.echo():
    weightEnc = WeightOfEvidenceEncoder()
    X_woe_nom = weightEnc.fit_transform(X_woe_nom,df['target'])
        
button6 = st.button('Apply Weight of Evidence Encoding')
if button6:
    st.dataframe(X_woe_nom)
    
X_woe_ord = df[ord_cols].copy()
X_woe_ord[['deg-malig']] = X_woe_ord[['deg-malig']].astype('object')
weightEnc = WeightOfEvidenceEncoder()
X_woe_ord = weightEnc.fit_transform(X_woe_ord,df['target'])
       

X_label = pd.read_csv('../data/X_label.csv')
X_ord = pd.read_csv('../data/X_ord.csv')
X_ohe_nom = pd.read_csv('../data/X_ohe_nom.csv')
X_ohe_ord = pd.read_csv('../data/X_ohe_ord.csv')
X_binary = pd.read_csv('../data/X_binary.csv')
X_base3 = pd.read_csv('../data/X_base3.csv')
X_thermo = pd.read_csv('../data/X_thermo.csv')

st.header('Now We Can Construct Logistic Regression Model for Classification')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

with st.echo():
    def logistic(X,y):
        model = LogisticRegression(C = 0.12345678987654321, solver = "lbfgs", max_iter = 5000, tol = 1e-2, n_jobs = 48)
        model.fit(X, y)
        score = cross_validate(model, X, y, cv=3, scoring="roc_auc")["test_score"].mean()
        print('AUC Score: ',f"{score:.6f}")
    
#st.header("Here Is The Results: ")