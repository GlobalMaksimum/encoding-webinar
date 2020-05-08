import streamlit as st
import pandas as pd
import numpy as np

def content():
    
    st.markdown('## What If Our String Feature Has an Order?')

    with st.echo():
        'Large' > 'Medium' > 'Small'
        'A' > 'B'
        '15-17' > '3-5'

    st.markdown("""
            There is an order between the string values that might need preserving.
             * Let's check our data for a such case.

            """)

    df = pd.read_csv('data/breast-cancer.csv')

    df.rename(columns={'Class': 'target'}, inplace=True)

    df.fillna('unknown', inplace = True)
    
    feat = st.selectbox('Select Feature',('age','tumor-size',
                                        'inv-nodes','deg-malig'))

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

    st.header(':question: Does Label Encoder Infer and Preserve the Relationship')
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
        
    with st.echo():
        df[feat].unique()

    st.markdown('Unique values are')
    st.write(df[feat].unique())
    
    button = st.button('Apply Label Encoder')

    if button:
        enc_le = le.fit(df[feat]).transform(df[feat].unique()).reshape(-1,1)
        st.subheader('Mapped Values')
        st.write(np.hstack([df[feat].unique().reshape(-1,1),enc_le]))
        st.subheader('Pairwise Distances')
        from sklearn.metrics import pairwise_distances
        with st.echo():
            pairwise_distances(enc_le.reshape(-1,1))
        st.write(pairwise_distances(enc_le.reshape(-1,1)))

        st.info('Encoded order is arbitrary. But we know our specific order...')


    st.header(':question: Can we inject the a specific order information to our encoder')
    
    st.subheader("`sklearn` has a special module for this: OrdinalEncoder")

    with st.echo():
        from sklearn.preprocessing import OrdinalEncoder
        
    df = pd.DataFrame(df[feat])
    X_ord = pd.DataFrame(df[feat])
    
    st.warning(":exclamation: We need to give order between categories.")
        
    showImplementation = st.checkbox('Show Code', key='Similarity1') 
    
    if showImplementation:
        with st.echo():
            if feat == 'age':
                oe = OrdinalEncoder(categories=[['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']])

            elif feat == 'tumor-size':
                 oe = OrdinalEncoder(categories=[['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54']])

            elif feat == 'inv-nodes':
                oe = OrdinalEncoder(categories=[['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '24-26']])

            elif feat == 'deg-malig':
                oe = OrdinalEncoder()
        
            X_ord[[feat]] = oe.fit_transform(df[[feat]])
            
    ##########
    
    if feat == 'age':
        oe = OrdinalEncoder(categories=[['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']])

    elif feat == 'tumor-size':
        oe = OrdinalEncoder(categories=[['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54']])

    elif feat == 'inv-nodes':
        oe = OrdinalEncoder(categories=[['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '24-26']])

    elif feat == 'deg-malig':
        oe = OrdinalEncoder()
        
    X_ord[[feat]] = oe.fit_transform(df[[feat]])
            
    button2 = st.button('Apply Ordinal Encoder')

    if button2:
        enc_oe = oe.fit(df[[feat]]).transform(df[[feat]].drop_duplicates()).reshape(-1,1)
        st.subheader('Mapped Values')
        mapped = np.hstack([df[feat].unique().reshape(-1,1),enc_oe])
        mapped_df = pd.DataFrame({'0':mapped[:,0],'1':mapped[:,1]})
        sorted_df = mapped_df.sort_values('1')
        sorted_df['1'] = sorted_df['1'].astype(np.int8)
        st.write(sorted_df)
        st.subheader('Pairwise Distances')
        from sklearn.metrics import pairwise_distances
        st.write(pairwise_distances(sorted_df['1'].values.reshape(-1,1)))
