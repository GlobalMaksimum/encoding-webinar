import streamlit as st
import pandas as pd
import numpy as np


st.header('What if our string feature has an Alphanumeric Order?')

with st.echo():
    'A' > 'B'
    'A' > 'a'
    '42' > '13'

st.markdown("""
        There is an order between the string values that might need preserving.
         * Let's check our data for a such case.
        
        """)

df = pd.read_csv('../data/breast-cancer.csv')

feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

st.dataframe(df[f'{feat}'])

if feat=='age':
    st.markdown("""
    `Age` seems like to display such order.
    """ )
elif feat=='inv-nodes':
    st.markdown("""
        So does `inv-nodes`. 
        """ )
elif feat=='tumor-size':
    st.markdown("""
        So does `tumor-size`. 
        """ )

st.header('Does Label Encoder Infer and Preserve the Relationship?')

with st.echo():
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    unq_values = df[feat].unique()
    unq_values



button = st.button('fit_transform label enc')

if button:
    enc_le = le.fit(df[feat]).transform(unq_values).reshape(-1,1)
    st.subheader('Mapped Values')
    st.write(np.hstack([unq_values.reshape(-1,1),enc_le]))
    st.subheader('Pairwise Distances')
    with st.echo():
        from  sklearn.metrics import pairwise_distances
        pairwise_distances(enc_le.reshape(-1,1))
    st.write(pairwise_distances(enc_le.reshape(-1,1)))

    st.markdown('Encoded order is arbitrary. But we know our specific order...')


st.header('Can we inject the a specific order information to our encoder.')

with st.echo():
    from sklearn.preprocessing import OrdinalEncoder
    oe = OrdinalEncoder(categories=[['0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44', '45-49','50-54']])


button2 = st.button('fit_transform ordinal')

if button2:
    enc_oe = oe.fit(df[[feat]]).transform(df[[feat]].drop_duplicates()).reshape(-1,1)
    st.subheader('Mapped Values')
    mapped = np.hstack([unq_values.reshape(-1,1),enc_oe])
    mapped_df = pd.DataFrame({'0':mapped[:,0],'1':mapped[:,1]})
    sorted_df = mapped_df.sort_values('1')
    sorted_df['1'] = sorted_df['1'].astype(np.int8)
    st.write(sorted_df)
    st.subheader('Pairwise Distances')
    with st.echo():
        from sklearn.metrics import pairwise_distances

    st.write(pairwise_distances(sorted_df['1'].values.reshape(-1,1)))
