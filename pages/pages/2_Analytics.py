import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📈 Health Analytics")

try:
    heart_df = pd.read_csv("heart.csv")
    stroke_df = pd.read_csv("healthcare-dataset-stroke-datas.csv")
except:
    st.warning("Dataset files not found.")
    st.stop()

st.subheader("Age Distribution (Heart Dataset)")
plt.figure()
plt.hist(heart_df["age"], bins=20)
st.pyplot(plt)

st.subheader("Stroke Cases Count")
plt.figure()
stroke_counts = stroke_df["stroke"].value_counts()
plt.bar(stroke_counts.index, stroke_counts.values)
st.pyplot(plt)
