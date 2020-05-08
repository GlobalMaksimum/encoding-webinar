import streamlit as st
from demo import *

chapters = ("Why Encoding?",
            "Dataset",
            "Label Encoding",
            "One Hot Encoding",
            "Ordinal Encoding",
            "Thermometer (Unary) Encoding",
            "Binary Encoding",
            "BaseN Encoding",
            "Frequency Encoding",
            "Target (Mean) Encoding",
            "K-Fold Target Encoding",
            "Weight of Evidence Encoding",
            "Cyclic Encoding")

callable_dict = {"Why Encoding?": intro,
            "Dataset": dset,
            "Label Encoding": lenc,
            "One Hot Encoding": ohe,
            "Ordinal Encoding": oenc,
            "Thermometer (Unary) Encoding": thermo,
            "Binary Encoding": bnry,
            "BaseN Encoding": base,
            "Frequency Encoding": freq,
            "Target (Mean) Encoding": trgt,
            "K-Fold Target Encoding": kfold,
            "Weight of Evidence Encoding": woe,
            "Cyclic Encoding": cyc
               }


st.sidebar.title("Agenda")

part = st.sidebar.radio("", chapters)

callable_dict.get(part, lambda: None)()