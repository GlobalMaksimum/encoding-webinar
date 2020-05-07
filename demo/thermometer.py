import streamlit as st
import pandas as pd
import numpy as np 
from PIL import Image

def content():
    
    df = pd.read_csv('data/breast-cancer.csv')
    
    df.rename(columns={'Class': 'target'}, inplace=True)
    df.fillna('unknown', inplace = True)
    
    st.header("Thermometer (Unary) Encoding")

    image2 = Image.open('images/Thermo.png')
    st.image(image2, use_column_width=True)

    st.markdown("For ordinal features by applying thermometer encoding we can preserve the order.")

    feat = st.selectbox('Select Feature',('age','tumor-size',
                                    'inv-nodes','deg-malig'))

    X_thermo = pd.DataFrame(df[feat])

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

        if feat == 'age':
            sort_key = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79'].index

        elif feat == 'tumor-size':
            sort_key = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54'].index

        elif feat == 'inv-nodes':
            sort_key = ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '24-26'].index

        elif feat == 'deg-malig':
            sort_key = int

        else:
            raise ValueError(feat)

        thermoEnc = ThermometerEncoder(sort_key = sort_key)
        thermos.append(thermoEnc.fit_transform(X_thermo[feat]))

    thermo_ohc = scipy.sparse.hstack(thermos).tocsr()
    thermo_ohc = scipy.sparse.csr_matrix(thermo_ohc).todense()
    X_thermo = pd.DataFrame(thermo_ohc)

    st.dataframe(df[feat])

    button5 = st.button('Apply Thermometer Encoding')
    if button5:
        st.dataframe(X_thermo)
