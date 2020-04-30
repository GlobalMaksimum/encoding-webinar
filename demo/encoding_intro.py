import streamlit as st
import pandas as pd 
import pydoc

def content():
    st.header('Why do we need encoding')
    st.markdown("""
            Because majority of algorithms are implemented using Linear Algebra primitives. Such as
            * Logistic Regression
            * SVM
            * LightGBM
    """)

    st.subheader('Example scikit-learn API')

    with st.echo():
        from sklearn.linear_model import LogisticRegression

    with st.echo():
        strhelp = pydoc.render_doc(LogisticRegression.fit)


    strhelp = pydoc.render_doc(LogisticRegression.fit)
    st.markdown(strhelp)


    st.subheader('A non-rigirous: How to convert numeric data into non numeric strings')
    with st.echo():
        df = pd.read_csv('data/breast-cancer.csv')
    feats = st.multiselect('Select a couple features',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

    if feats:
        st.dataframe(df[feats])

    st.subheader('Fail to Build a LogisticRegression Model using String Values')

    dt = st.selectbox('Select Data Type:', ('string','somethig else'))

    if dt=='string':
        st.markdown("""
                    ```python
                    lr = LogisticRegression()
                    lr.fit(df[feats],df.Class)
                    ```
        """)
        with st.echo():
            lr = LogisticRegression()
            lr.fit(df[feats],df.Class)

    else:
        st.subheader('Simplest Idea')
        st.markdown(""" 
        What if I simply replace string value with arbitrarily picked unique values (aka dictionary mapping)
        * Unique value per column/feature/attribute
        """)

        with st.echo():
            X_enc = df[['breast', 'irradiat']].replace({'right':1,'left':0, 'no':0,'yes':1})
            y_enc = df.Class.replace( {'recurrence-events':1,'no-recurrence-events':0 } )

            lr = LogisticRegression().fit(X_enc, y_enc)


    st.success('IT WORKED!!')
    st.markdown("""
        * Can we have better encoding techniques ?
        * That's why we are here
    """)
    
if __name__ == '__main__':
    content()