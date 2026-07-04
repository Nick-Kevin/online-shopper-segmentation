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

# list of columns in the dataframe before scaling
preprocessed_features = [
    "Administrative", "Administrative_Duration", "Informational",
    "Informational_Duration", "ProductRelated_Duration",
    "ExitRates", "PageValues", "SpecialDay", "Weekend",
    "VisitorType_New_Visitor", "VisitorType_Other",
    "VisitorType_Returning_Visitor", "Month_Aug", "Month_Dec",
    "Month_Feb", "Month_Jul", "Month_June", "Month_Mar", "Month_May",
    "Month_Nov", "Month_Oct", "Month_Sep"
]

def preprare_input(data, feature_list):
    """
        Make the input data similar to the preprocessed data before scaling during the model implementation

        Args:
        - data: the input data by the user
        - feature_list: the list of features required for the rest of the process

        Return: dataframe similar in the preprocessing stage
    """

    input_data = {feature: data.get(feature, 0) for feature in feature_list}
    return pd.DataFrame(data=input_data, index=[0])

def define_label(label_num):
    """
        get the label of the predicted cluster number that the gmm model has predicted

        Arg:
            - label_num: the cluster number predicted
        Return:
            - the explained number
    """

    match label_num:
        case 0:
            label = "serious shopper"
        case 1:
            label = "new visitor with high purchase intention"
        case 2:
            label = "casual visitor"
        case _:
            label = "unkown"
    return label


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
    date = st.date_input("Date of the session")
    special_day = st.selectbox(
        "Is a special day?",
        ["No", "Yes"],
        help="Is the browsing date close to special days or holidays (eg Mother's Day)"
    )
    st.write("---")
    st.markdown("""
        Connect with me on [LinkedIn](https://www.linkedin.com/in/nick-kevin-razafinirina-988b34248/),
        [GitHub](https://github.com/Nick-Kevin)
    """)
    st.markdown("""
        If you like this demo, please consider giving it a ⭐ on
        [GitHub](https://github.com/Nick-Kevin/online-shopper-segmentation)
    """)

# ------- MAIN PANEL: USER BEHAVIORAL DATA -------
st.markdown("## 📊 Live Session Behavior")

# Using a 3-column grid of inputs
col11, col21, col31 = st.columns(3)

with col11:
    st.markdown("### 🗺️ Administrative pages")
    admin_page = st.number_input(
        "Page visited",
        min_value=0,
        value=2,
        step=1,
        help="number of administrative type pages that the user visited"
    )
    admin_duration = st.number_input(
        "Time spent (seconds)",
        min_value=0,
        value=385,
        help="amount of time spent by the visitor in administrative pages category"
    )

with col21:
    st.markdown("### ℹ️ Informational pages")
    info_page = st.number_input(
        "Page number",
        min_value=0,
        value=0,
        step=1,
        help="number of informational type pages that the user visited"
    )
    info_duration = st.number_input(
        "Time spent (seconds)",
        min_value=0,
        value=0,
        help="amount of time spent by the visitor in informational pages category"
    )

with col31:
    st.markdown("### 🏷️ Product related pages")
    product_page = st.number_input(
        "Page number",
        min_value=0,
        value=1,
        step=1,
        help="number of product related type pages that the user visited"
    )
    product_duration = st.number_input(
        "Time spent (seconds)",
        min_value=0,
        value=75,
        help="amount of time spent by the visitor in product related pages category"
    )

st.write("---")

st.markdown("## 📈 Engagement metrics (Google Analytics)")
col12, col22, col32 = st.columns(3)

with col12:
    bounce_rate = st.slider("Bounce Rate", min_value=0.0, max_value=0.2, value=0.07, format="%0.2f")
with col22:
    exit_rate = st.slider("Exit rate", min_value=0.0, max_value=0.2, value=0.04, format="%0.2f")
with col32:
    page_value = st.number_input(
        "Page value ($)",
        min_value=0,
        value=15,
        help="Average value for a web page that a user has visited before completing an e-commerce transaction."
    )

st.write("---")

st.markdown("## 🔮 Prediction output")

if st.button("Estimate session cluster", type="primary"):
    # set weekend to 1 if weekend, 0 otherwise
    date_of_week = pd.to_datetime(date).day >= 5
    weekend = 1 if date_of_week else 0

    special_day = 1 if special_day == "Yes" else 0

    # format the visitor type for encoding
    match visitor_type:
        case "Returning":
            formatted_visitor_type = "Returning_Visitor"
        case "New":
            formatted_visitor_type = "New_Visitor"
        case "Other", _:
            formatted_visitor_type = "Other"

    month = pd.to_datetime(date).month

    match month:
        case 1,2:
            formatted_month = "Feb"
        case 3:
            formatted_month = "Mar"
        case 4, 5:
            formatted_month = "May"
        case 6:
            formatted_month = "June"
        case 7:
            formatted_month = "Jul"
        case 8:
            formatted_month = "Aug"
        case 9:
            formatted_month = "Sep"
        case 10:
            formatted_month = "Oct"
        case 11:
            formatted_month = "Nov"
        case 12, _:
            formatted_month = "Dec"

    user_data = {
        "Administrative": admin_page,
        "Administrative_Duration": admin_duration,
        "Informational": info_page,
        "Informational_Duration": info_duration,
        "ProductRelated_Duration": product_duration,
        "ExitRates": exit_rate,
        "PageValues": page_value,
        "SpecialDay": special_day,
        "Weekend": weekend,
        f"VisitorType_{formatted_visitor_type}": 1,
        f"Month_{formatted_month}": 1
    }

    input_df = preprare_input(user_data, preprocessed_features)
    input_pca = pca2.transform(input_df)
    cluster_num = gmm_model.predict(input_pca)
    LABEL = define_label(cluster_num)

    if LABEL == "unkown":
        st.error('Error! The model cannot estimate the cluster.', icon="🚨")
    else:
        st.success(f"The machine learning model estimates that the session indicates a \n{LABEL}.")
        st.balloons()
