import streamlit as st
import pandas as pd
import numpy as np 
import altair as alt

def content():
    st.subheader('Data Import')

    st.markdown('* Dataset download source from [OpenML Breast cancer](https://www.openml.org/d/13)')
    st.markdown('* This breast cancer domain was obtained from the University Medical Centre, Institute of Oncology, Ljubljana, Yugoslavia.')
    st.markdown('* ***Number of Instances:*** 286')
    st.markdown('* ***Number of Features:*** 9')
    st.markdown('* ***Target Class:*** Whether the tumor will recur or not.')
    st.markdown("""
                * Data fields (Features)  
                    * **age:** Age (in years at last birthday) of the patient at the time of diagnosis 
                    * **menopause:** Whether the patient is pre- or postmenopausal at time of diagnosis       
                    * **tumor-size:** The greatest diameter (in mm) of the excised tumor       
                    * **inv-nodes:** The number (range 0 - 39) of axillary lymph nodes that contain metastatic breast cancer visible on histological examination         
                    * **node-caps:** If the cancer does metastasise to a lymph node, although outside the original site of the tumor it may remain “contained” by the capsule of the lymph node. However, over time, and with more aggressive disease, the tumor may replace the lymph node and then penetrate the capsule, allowing it to invade the surrounding tissues   
                    * **deg-malig:**  Degree of malignancy
                    * **breast:** Breast cancer may obviously occur in either breast
                    * **breast-quad:** The breast may be divided into four quadrants, using the nipple as a central point
                    * **irradiat:**  Radiation therapy is a treatment that uses high-energy x-rays to destroy cancer cells.    
                """)  


    with st.echo():
        df = pd.read_csv('../data/breast-cancer.csv')

    st.dataframe(df)

    st.subheader('Simple Exploration')

    feat = st.selectbox('Select Feature',('age','menopause','tumor-size',
                                    'inv-nodes','node-caps','deg-malig',
                                    'breast','breast-quad','irradiat'))

    bar1 = alt.Chart(df).mark_bar().encode(
        x = alt.X(feat),
        y = 'count()',
        color = 'Class'
    )

    st.altair_chart(bar1,use_container_width=True)


if __name__ == '__main__':
    content()