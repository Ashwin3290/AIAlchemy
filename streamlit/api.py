import pickle 
import numpy as np
import warnings
import time
import os
import random
import pandas as pd
from pycaret.regression import RegressionExperiment
import pycaret.classification as ClassificationExperiment
from datetime import datetime
import io
from matplotlib import pyplot as plt
from pycaret.utils import check_metric
from sklearn.model_selection import train_test_split
from pycaret.classification import setup, compare_models, finalize_model, save_model, predict_model

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'xlsx', 'csv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_file(file_path):
    if allowed_file(file_path):
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Invalid file format")
        return df
    else:
        raise ValueError("Invalid file format")

def regression(df, target):
    rgr = RegressionExperiment()
    rgr.setup(df, target=target)
    best_model = rgr.compare_models()
    compare_df = rgr.pull()
    rgr.finalize_model(best_model)
    pipeline = pickle.dumps(rgr.save_model(best_model, model_name='best_model'))
    os.remove("best_model.pkl")
    model_comparison = compare_df.to_dict()
    compare_df = pickle.dumps(compare_df)
    return {"status":True,"message":"Model created successfully","model_comparison":model_comparison}

def classification(df, target):
    clf = ClassificationExperiment
    clf.setup(df, target=target)
    best_model = clf.compare_models()
    compare_df = clf.pull()
    clf.finalize_model(best_model)
    pipeline = pickle.dumps(clf.save_model(best_model, model_name='best_model'))
    os.remove("best_model.pkl")
    model_comparison = compare_df.to_dict()
    compare_df = pickle.dumps(compare_df)
    return {"status":True,"message":"Model created successfully","model_comparison":model_comparison}

def get_plots(pipeline):
    plt.figure(figsize=(10, 8))
    plot_model(pipeline, plot='feature')
    plt.tight_layout()
    plt.show()

def get_cluster_plots(pipeline):
    plt.figure(figsize=(10, 8))
    plot_model(pipeline)
    plt.tight_layout()
    plt.show()

def main(file_path, target, option):
    df = read_file(file_path)
    if option == 'regression':
        result = regression(df, target)
        pipeline = pickle.loads(result['model_comparison'])
        get_plots(pipeline)
        get_cluster_plots(pipeline)
    elif option == 'classification':
        result = classification(df, target)
        pipeline = pickle.loads(result['model_comparison'])
        get_plots(pipeline)
        get_cluster_plots(pipeline)
    else:
        raise ValueError("Invalid option")

if __name__ == '__main__':
    file_path = 'your_file_path'
    target = 'your_target_column'
    option = 'regression'  # or 'classification'
    main(file_path, target, option)


    from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoNickML")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target, silent=True)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")