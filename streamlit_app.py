import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
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
            toefl = st.number_input('TOEFL score',max_value=120, value= 100, step=1)
        with filter5:
            gre = st.number_input('GRE score',max_value=340, value=300, step=10)
        with filter6:
            gpa = st.number_input('GPA out of 10',max_value=10.,step=0.5, value=7.0)
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
    proba = float(max(0,proba))

    # outcome_df

    col1, col2= st.columns(2)
    with col1:
        st.write(f'Because of self-censorship you will most likely apply to **Tier {uni_tier}** colleges')
    with col2:
        st.metric(f'Probability of getting in a Tier {uni_tier} University', "{:.0%}".format(proba))
        scores = [proba,1-proba] 
        labels=['A','B'] 
        marker_colors = ['purple','snow']
        fig = go.Figure(data=[go.Pie(values=scores, labels=labels, direction='clockwise', hole=0.5,marker_colors = marker_colors, sort=False)])
        fig.update_traces(textinfo='none')
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, width=200,height=200, showlegend=False)
        st.plotly_chart(fig,use_container_width=False)           


if page =='Model':
    df = load_csv()
    st.header('Some storytelling')
    st.write('#### The University Tier feature is problematic, and not usable because of self-censoship, so I decided to use a **KNN classifier** to infer it from other features')
    st.write('Table: Average values grouped by University Rating: self-censorship much...?')
    st.write(df.groupby(['University Rating']).mean())
    st.write('#### Part 1: Using classification to model self-censorship')

    models = ['K-NN','Naive Bayes Classifier','Decision Tree','Random Forest','AdaBoost']
    models_sk = [KNeighborsClassifier(n_neighbors=10),GaussianNB(),DecisionTreeClassifier(),RandomForestClassifier(),AdaBoostClassifier()]
    model_dict = dict(zip(models,models_sk))
    X_tier = df.drop(['University Rating','Chance of Admit '], axis=1)
    Y_tier = df['University Rating']
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox('Select Classifier',models)
    with col2:
        test_size = st.slider('Test size',0.05,1.,step=0.05,value=0.2)
    classifier = model_dict[model]
    X_train, X_test, y_train, y_test = train_test_split(X_tier, Y_tier,test_size = test_size, random_state=42)
    classifier.fit(X_train.values, y_train.values)
    st.write(f" Model accuracy **{classifier.score(X_test.values, y_test.values):.3f}**")
    
    with st.form('Test the model'):
        filter1, filter2, filter3, filter4, filter5, filter6 = st.columns(6)
        with filter1:
            sop = st.slider('SOP strength',1,5,step=1)
        with filter2:
            research = st.checkbox('Research Experience?')
        with filter3:
            lor = st.slider('LORs strength',1,5,step=1)
        with filter4:
            toefl = st.number_input('TOEFL score',max_value=120, value= 100, step=1)
        with filter5:
            gre = st.number_input('GRE score',max_value=340, value=300, step=10)
        with filter6:
            gpa = st.number_input('GPA out of 10',max_value=10.,step=0.5, value=7.0)
        submitted = st.form_submit_button('Predict University Rating')
        
    uni_tier = classifier.predict(np.array([gre, toefl, sop, lor, gpa, research]).reshape(1,-1))
    st.metric('Predicted Rating /5 (higher is better)',int(uni_tier))


    st.write('#### Part 2: Predicting Chance of Admit')
    st.write('##### Start simple:  Linear Regression... cannot be used')
    
    col1, col2 = st.columns(2)
    with col1:
        st.write('Multi-collinearity')
        fig = px.imshow(df.drop(['Chance of Admit '],axis=1).corr(), text_auto=True)
        st.plotly_chart(fig, use_container_width=False)
    
    with col2:
        X = st.multiselect('Select Regressors', df.columns, ['CGPA'])
        Y = df['Chance of Admit ']
        model = sm.OLS(Y, sm.add_constant(df[X]))
        results = model.fit()
        y_hat = results.predict()
        st.write(f'A regression on {str(X)} leads to an **R-squared of {np.round(100*results.rsquared_adj)}%** ')

        with st.expander('Show Regression Output'):
            st.write(results.summary(slim=True))
        

        fig = px.scatter(x=y_hat ,y = Y, trendline='ols')
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',}) 
        fig.update_layout(showlegend=True, xaxis_title='Predicted')
        st.plotly_chart(fig)

    st.write('##### Given how multicollinear the data is, I decided to use **Projection methods** like **Partial Least Squares** & **Principal Components Regression**')
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider('Select test size',0.05,1.0,step=0.05, value=0.2)
        n_components = st.slider('Number of components to keep',1,6, value=3)
    X = df.drop(['Chance of Admit ', 'University Rating'], axis=1)
    Y = df['Chance of Admit ']
    X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size = test_size, random_state=42)

    pcr = make_pipeline(StandardScaler(), PCA(n_components=n_components), LinearRegression())
    pcr.fit(X_train, y_train)
    pca = pcr.named_steps["pca"]  # retrieve the PCA step of the pipeline
    ols = pcr.named_steps["linearregression"]
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)
    y_hat_pls = pls.predict(X_test).reshape(len(y_test))
    y_hat_pcr = pcr.predict(X_test).reshape(len(y_test))
    with col1:
        st.write('**Model Results & Coefficients**')
        st.write(f"PCR model R2: **{pcr.score(X_test, y_test):.3f}**")
        st.write(f"PLS model R2: **{pls.score(X_test, y_test):.3f}**")
        coefs_pls = pls.coef_.reshape(len(X.T))
        coefs_pcr = pca.inverse_transform(ols.coef_).T
        coefs = pd.DataFrame(list(zip(coefs_pls,coefs_pcr)), columns = ['PLS','PCR'],index=X.columns)
        fig = px.bar(coefs, barmode="group")
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',}) 
        st.plotly_chart(fig)
    

    with col2:
        df = pd.DataFrame(list(zip(y_test, y_hat_pls, y_hat_pcr)), columns=['Actual','PLS','PCR'])
        fig = px.scatter(df,x='Actual',y=['PLS','PCR'])
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',}) 
        fig.update_layout(showlegend=True, xaxis_title='actual')
        st.write(f'**PCR & PLS results with {n_components} components**')

        st.plotly_chart(fig)
        st.write('#### No surprise, CGPA is the most important feature...')
        gpa_coef = np.round(100*coefs.loc['CGPA','PLS'])
        st.write(f'##### According to the PLS model, a 1 point increase in GPA leads to a {gpa_coef} % increase in chance of admission')
    
    