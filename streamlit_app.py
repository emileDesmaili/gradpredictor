import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import pickle

# PAGE SETUP
st.set_page_config(
    page_title="GradPredictor",
    layout="wide",
    page_icon="streamlit_assets/assets/icon_logo.PNG",

)


# From https://discuss.streamlit.io/t/how-to-center-images-latex-header-title-etc/1946/4
with open("streamlit_assets/style.css") as f:
    st.markdown("""<link href='http://fonts.googleapis.com/css?family=Roboto:400,100,100italic,300,300italic,400italic,500,500italic,700,700italic,900italic,900' rel='stylesheet' type='text/css'>""", unsafe_allow_html=True)
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


left_column, center_column,right_column = st.columns([1,3,1])

with left_column:
    st.info("Project using streamlit")
with right_column:
    st.write("##### Authors\nThis tool has been developed by [Emile D. Esmaili](https://github.com/emileDesmaili)")
with center_column:
    st.image("streamlit_assets/assets/app_logo.PNG")



st.sidebar.write(f'# Welcome')

page_container = st.sidebar.container()
with page_container:
    page = option_menu("Menu", ["ChanceMe!", "Model"], 
    icons=['reddit','dpad'], menu_icon="cast", default_index=0)


@st.cache()
def load_csv():
    return pd.read_csv('data/admission_data.csv')
@st.cache():

@st.cache(allow_output_mutation=True)
def load_predictor():
    filename = 'models/pls_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

data = load_csv()

if page =="ChanceMe!":
    model = load_predictor()
    
