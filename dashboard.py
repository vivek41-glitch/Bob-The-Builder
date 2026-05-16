import streamlit as st
import pandas as pd
import plotly.express as px
from config import EXPERIMENT_LOG_FILE
import os

st.set_page_config(page_title="Bob The Builder", layout="wide")
st.title("Bob The Builder - AutoDL Optimizer")
st.subheader("IBM Bob as a Senior Deep Learning Engineer")

with st.sidebar:
    st.header("Experiment Info")
    if os.path.exists(EXPERIMENT_LOG_FILE):
        df = pd.read_csv(EXPERIMENT_LOG_FILE)
        st.metric("Total Experiments", len(df))
        if len(df) > 0:
            best_row = df.loc[df['score'].idxmax()]
            st.metric("Best Accuracy", f"{best_row['accuracy']:.2f}%")
            baseline_row = df[df['iteration'] == 'baseline']
            if not baseline_row.empty:
                improvement = best_row['accuracy'] - baseline_row.iloc[0]['accuracy']
                st.metric("Improvement", f"{improvement:+.2f}%")

col1, col2 = st.columns(2)

with col1:
    st.header("Accuracy Progress")
    if os.path.exists(EXPERIMENT_LOG_FILE):
        df = pd.read_csv(EXPERIMENT_LOG_FILE)
        # Create iteration number column based on row index
        df['iteration_num'] = range(1, len(df) + 1)
        fig = px.line(df, x='iteration_num', y='accuracy', title='Accuracy Over Iterations', markers=True)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("Score Progress")
    if os.path.exists(EXPERIMENT_LOG_FILE):
        df = pd.read_csv(EXPERIMENT_LOG_FILE)
        df['iteration_num'] = range(1, len(df) + 1)
        fig = px.line(df, x='iteration_num', y='score', title='Score Over Iterations', markers=True)
        st.plotly_chart(fig, use_container_width=True)

st.header("Experiment History")
if os.path.exists(EXPERIMENT_LOG_FILE):
    df = pd.read_csv(EXPERIMENT_LOG_FILE)
    st.dataframe(df, use_container_width=True)

st.header("Best Configuration")
if os.path.exists(EXPERIMENT_LOG_FILE):
    df = pd.read_csv(EXPERIMENT_LOG_FILE)
    best_row = df.loc[df['score'].idxmax()]
    st.json(best_row.to_dict())
    