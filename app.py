import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_model():
    return pickle.load(open("components/gmm-model.pkl", "rb"))

@st.cache_resource
def load_encoder():
    return pickle.load(open("components/one-hot-encoder.pkl", "rb"))

@st.cache_resource
def load_pca():
    return pickle.load(open("components/pca2.pkl", "rb"))

@st.cache_resource
def load_scaler():
    return pickle.load(open("components/scaler.pkl", "rb"))

gmm_model = load_model()
encoder = load_encoder()
pca2 = load_pca()
scaler = load_scaler()

# page configuration
st.set_page_config(
    page_title="Cluster estimation of online shopper",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)
