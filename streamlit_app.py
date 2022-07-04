import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import math

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

@st.cache(allow_output_mutation=True)
def load_predictor():
    filename = 'models/pls_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

@st.cache(allow_output_mutation=True)
def load_classifier():
    filename = 'models/class_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

data = load_csv()

if page =="ChanceMe!":
    model = load_predictor()
    classifier = load_classifier()

    with st.form('Chancer'):
        filter1, filter2, filter3, filter4, filter5, filter6 = st.columns(6)
        with filter1:
            sop = st.slider('SOP strength',1,5,step=1)
        with filter2:
            research = st.checkbox('Research Experience?')
        with filter3:
            lor = st.slider('LORs strength',1,5,step=1)
        with filter4:
            toefl = st.number_input('TOEFL score',max_value=120,step=1)
        with filter5:
            gre = st.number_input('GRE score',max_value=340,step=10)
        with filter6:
            gpa = st.number_input('GPA out of 10',max_value=10.,step=0.01)
        
        
        submitted = st.form_submit_button('ChanceMe!')
    uni_tier = classifier.predict(np.array([gre, toefl, sop, lor, gpa, research]).reshape(1,-1))
    uni_tier = int(uni_tier)
    if uni_tier == 5:
        uni_tier = 1
    elif uni_tier == 4:
        uni_tier = 2
    elif uni_tier == 2:
        uni_tier = 4
    elif uni_tier == 1:
        uni_tier = 5
    proba = model.predict(np.array([gre, toefl, int(uni_tier), sop, lor, gpa, research]).reshape(1,-1))

    # outcome_df

    col1, col2= st.columns(2)
    with col1:
        st.write(f'Because of Self-censorship you will apply to Tier {uni_tier} colleges')
    with col2:
        st.metric(f'Probability of getting in a Tier {uni_tier} University', np.round(proba,3))



