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

# ------- APP HEADER -------
st.title("🛍️ Online Shopper Purchasing Intent Segmetation")
st.markdown("""
    This dashboard uses a real-time e-commerce session to estimate the characteristic of visitors.
    Adjust the behavior metrics below to see the estimation update.
""")
st.write("---")

# ------- SIDEBAR -------
st.sidebar.header("👤 Visitor and time context")

with st.sidebar:
    visitor_type = st.selectbox("Visitor Type", ["Returning", "New", "Other"])
    date = st.date_input("Date")

# ------- MAIN PANEL: USER BEHAVIORAL DATA -------
st.markdown("## 📊 Live Session Behavior")

# Using a 3-column grid of inputs
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🗺️ Administrative pages")
    admin_page = st.number_input("Page visited", min_value=0, value=2, step=1)
    admin_duration = st.number_input("Time spent (seconds)", min_value=0, value=385)

with col2:
    st.markdown("### ℹ️ Informational pages")
    info_page = st.number_input("Page number", min_value=0, value=0, step=1)
    info_duration = st.number_input("Time spent (seconds)", min_value=0, value=0)

with col3:
    st.markdown("### 🏷️ Product related pages")
    product_page = st.number_input("Page number", min_value=0, value=1, step=1)
    product_duration = st.number_input("Time spent (seconds)", min_value=0, value=75)
