import streamlit as st
import pandas as pd 
import pydoc
from PIL import Image

def content():
    
    st.title("A Quick Look at Encoding")

    st.markdown("""
              Encoding is the process of converting the data or a given sequence of characters, symbols, alphabets etc., into a specified format, for the secured transmission of data.
            """)

    st.header('We Usually Have Categorical Variables in Our Dataset')

    st.markdown(' * A categorical variable is a variable that can take on one of a limited, and usually fixed number of possible values.')

    st.markdown(" * Categorical Variables can be divided into 2 categories:")

    image1 = Image.open('images/cat.png')

    st.image(image1, use_column_width=True)

    st.info(':pushpin:  Nominal variables have no numerical importance, besides ordinal variables have some order.')

    st.warning("""
    ### :exclamation: Why do we need encoding for Categorical Variables?
    Because majority of algorithms are implemented using Linear Algebra primitives. Such as:
    * Logistic Regression
    * SVM
    * LightGBM
        """)


    st.subheader('Example `sklearn` API')

    with st.echo():
        from sklearn.linear_model import LogisticRegression

    with st.echo():
        strhelp = pydoc.render_doc(LogisticRegression.fit)


    strhelp = pydoc.render_doc(LogisticRegression.fit)
    st.markdown(strhelp)


    st.subheader('A non-rigorous: How to convert non-numeric strings into numeric data?')

    with st.echo():
        df = pd.read_csv('data/breast-cancer.csv')
        
    df.rename(columns={'Class': 'target'}, inplace=True)
    df.fillna('unknown', inplace = True)
    
    feats = st.multiselect('Select a couple features',('deg-malig',
                                        'breast', 'irradiat'))

    if feats:
        st.dataframe(df[feats])

    dt = st.selectbox('Select Data Type:', ('string','numerical value'))

    st.subheader('Fit Logistic Regression Model')
    
    if dt=='string':
        st.markdown("""
                        ```python
                        lr = LogisticRegression()
                        lr.fit(df[feats],df.Class)
                        ```
            """)
        st.warning(':exclamation: Fail to Build a LogisticRegression Model using Non-numeric Values')
        with st.echo():
            lr = LogisticRegression()
            lr.fit(df[feats],df.target)
    else:
        st.subheader('Simplest Idea')
        st.markdown(""" 
        What if I simply replace string value with arbitrarily picked unique values (aka dictionary mapping)
            """)
        st.markdown(""" 
            * Unique value per column/feature/attribute
            """)
        with st.echo():
            X_enc = df[['breast', 'irradiat']].replace({'right':1,'left':0, 'no':0,'yes':1})
            y_enc = df.target.replace( {'recurrence-events':1,'no-recurrence-events':0 } )

            lr = LogisticRegression().fit(X_enc, y_enc)

    st.success('IT WORKED :white_check_mark:')
    st.markdown("""
        * Can we have better encoding techniques ?
        * That's why we are here :tada::tada:
        """)