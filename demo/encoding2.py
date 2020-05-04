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

st.header("Another Encoding Technique is: One Hot Encoding")

st.markdown("In this method, we map each category to a vector that contains 1 and 0 denoting the presence or absence of the feature.")

image1 = Image.open('../images/ohe2.png')
st.image(image1, use_column_width=True)

st.markdown("This method produces a lot of columns that slows down the learning significantly if the number of the category is very high for the feature.")

st.markdown("Here we apply one hot encoding for nominal features.")

with st.echo():
    X_ohe_nom = df[nom_cols].copy()
    
st.dataframe(X_ohe_nom.head(10))

with st.echo():
    X_ohe_nom = pd.get_dummies(X_ohe_nom)

button1 = st.button('Apply One Hot Encoding')
if button1:
    st.dataframe(X_ohe_nom)
    
X_ohe_ord = df[ord_cols].copy()
X_ohe_ord[['deg-malig']] = X_ohe_ord[['deg-malig']].astype('object')
X_ohe_ord = pd.get_dummies(X_ohe_ord)

st.markdown("If we apply one hot encoding for ordinal features we can lose the order between categories.")

st.header("Here is The Solution: Thermometer (Unary) Encoding")

image2 = Image.open('../images/thermo3.png')
st.image(image2, use_column_width=True)

st.markdown("For ordinal features by applying thermometer encoding we can preserve the order.")

with st.echo():
    X_thermo = df[ord_cols].copy()
    
st.dataframe(X_thermo.head(10))

from sklearn.base import TransformerMixin
from itertools import repeat
import scipy

class ThermometerEncoder(TransformerMixin):
    """
    Assumes all values are known at fit
    """
    def __init__(self, sort_key=None):
        self.sort_key = sort_key
        self.value_map_ = None
    
    def fit(self, X, y=None):
        self.value_map_ = {val: i for i, val in enumerate(sorted(X.unique(), key=self.sort_key))}
        return self
    
    def transform(self, X, y=None):
        values = X.map(self.value_map_)
        
        possible_values = sorted(self.value_map_.values())
        
        idx1 = []
        idx2 = []
        
        all_indices = np.arange(len(X))
        
        for idx, val in enumerate(possible_values[:-1]):
            new_idxs = all_indices[values > val]
            idx1.extend(new_idxs)
            idx2.extend(repeat(idx, len(new_idxs)))
            
        result = scipy.sparse.coo_matrix(([1] * len(idx1), (idx1, idx2)), shape=(len(X), len(possible_values)), dtype="int8")
            
        return result

st.markdown("As in ordinal encoding we need to give order between categories.")
    
with st.echo():
    thermos = []

    for col in ord_cols:
        
        if col == 'age':
            sort_key = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79'].index
        
        elif col == 'tumor-size':
            sort_key = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54'].index
        
        elif col == 'inv-nodes':
            sort_key = ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '24-26'].index
        
        elif col == 'deg-malig':
            sort_key = int
        
        else:
            raise ValueError(col)
    
    thermoEnc = ThermometerEncoder(sort_key = sort_key)
    thermos.append(thermoEnc.fit_transform(X_thermo[col]))

thermo_ohc = scipy.sparse.hstack(thermos).tocsr()
thermo_ohc = scipy.sparse.csr_matrix(thermo_ohc).todense()
X_thermo = pd.DataFrame(thermo_ohc)

button5 = st.button('Apply Thermometer Encoding')
if button5:
    st.dataframe(X_thermo)

st.markdown("For example the first 10 samples of the 'age' feature are as in below in our dataset.")
    
st.dataframe(df['age'].head(10))

st.markdown("After applying thermometer encoding it looks like this:")

st.dataframe(X_thermo.iloc[:,:6].head(10))

st.header("We Can Reduce Dimensionality Using Binary Encoding")

st.markdown("Binary encoding converts a category into binary digits. Each binary digit creates one feature column.")

image3 = Image.open('../images/binary.png')
st.image(image3, use_column_width=True)

st.markdown("Compared to one hot encoding, this will require fewer feature columns.")

with st.echo():
    import category_encoders as ce
    
st.markdown("Here we apply binary encoding for ordinal columns.")
    
with st.echo():
    X_binary = df[ord_cols].copy()
    
st.dataframe(X_binary.head(10))

X_binary[['deg-malig']] = X_binary[['deg-malig']].astype('object')

with st.echo():
    binaryEnc = ce.BinaryEncoder()
    X_binary = binaryEnc.fit_transform(X_binary)

button3 = st.button('Apply Binary Encoding')
if button3:
    st.dataframe(X_binary)

st.header("We Can Reduce Dimensionality Even More by Applying BaseN Encoding")

st.markdown("BaseN encoding converts a category into base N digits.")

st.markdown("By reducing dimensionality, we also lose some information.")

st.markdown("Here we apply base 3 encoding for ordinal columns.")

with st.echo():
    X_base3 = df[ord_cols].copy()
    
st.dataframe(X_base3.head(10))

X_base3[['deg-malig']] = X_base3[['deg-malig']].astype('object')

with st.echo():
    base3Enc = ce.BaseNEncoder(base = 3)
    X_base3 = base3Enc.fit_transform(X_base3)

button4 = st.button('Apply Base3 Encoding')
if button4:
    st.dataframe(X_base3)

X_ohe_nom.to_csv('X_ohe_nom.csv', index=False)
X_ohe_ord.to_csv('X_ohe_ord.csv', index=False)
X_binary.to_csv('X_binary.csv', index=False)
X_base3.to_csv('X_base3.csv', index=False)
X_thermo.to_csv('X_thermo.csv', index=False)