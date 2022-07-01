import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
import numpy as np

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

data = load_csv()

if page =="ChanceMe!":
    model = load_predictor()

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
    proba_t5 = model.predict(np.array([gre, toefl, 1, sop, lor, gpa, research]).reshape(1,-1))
    proba_t4 = model.predict(np.array([gre, toefl, 2, sop, lor, gpa, research]).reshape(1,-1))
    proba_t3 = model.predict(np.array([gre, toefl, 3, sop, lor, gpa, research]).reshape(1,-1))
    proba_t2 = model.predict(np.array([gre, toefl, 4, sop, lor, gpa, research]).reshape(1,-1))
    proba_t1 = model.predict(np.array([gre, toefl, 5, sop, lor, gpa, research]).reshape(1,-1))
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric('Probability of getting in a Tier 1 University', np.round(proba_t1,3))
    with col2:
        st.metric('Probability of getting in a Tier 2 University', np.round(proba_t2,3))
    with col3:
        st.metric('Probability of getting in a Tier 3 University', np.round(proba_t3,3))
    with col4:
        st.metric('Probability of getting in a Tier 4 University', np.round(proba_t4,3))
    with col5:
        st.metric('Probability of getting in a Tier 5 University', np.round(proba_t5,3))


