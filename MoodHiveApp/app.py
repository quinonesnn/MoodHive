import streamlit as st
import pandas as pd
import numpy as np
import joblib


def main():
    st.title("MoodHive")
    menu = ["Home", "Details"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("How was your day today?")

        with st.form(key="emotionTextForm"):
            rawText = st.text_area("Begin typing...")
            submitTextBtn = st.form_submit_button(label="Submit")

        if submitTextBtn:
            col1, col2 = st.columns(2)
            with col1:
                st.success("Original Text")
                st.write(rawText)
                st.success("Prediction")

            with col2:
                st.success("Prediction Probability")

    elif choice == "Details":
        st.subheader("Details")


if __name__ == '__main__':
    main()
