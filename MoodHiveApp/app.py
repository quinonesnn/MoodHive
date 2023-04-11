import streamlit as st
import pandas as pd
import numpy as np
import chatgpt
from ModelPredictions import predict_samples
from ModelPredictions import preprocess_corpus
from ModelPredictions import BERTmodel
from gensim.summarization import keywords


def extract_keywords(text, num_keywords=5):
    return keywords(text, words=num_keywords).split('\n')


def get_selected_emotions(emotion1, emotion2, emotion3, predicted_emotions):
    selected_emotions = []
    if emotion1:
        selected_emotions.append(predicted_emotions.iloc[0, 0])
    if emotion2:
        selected_emotions.append(predicted_emotions.iloc[1, 0])
    if emotion3:
        selected_emotions.append(predicted_emotions.iloc[2, 0])
    return selected_emotions


def main():
    st.set_page_config(layout="wide")
    st.image("/Users/nickq/Repos/MoodHive/MoodHiveLogo.png", width=400)
    menu = ["Home", "Details"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":

        st.subheader("How was your day today?")

        with st.form(key="emotionTextForm"):
            rawText = st.text_area("Begin typing...")
            submitJournalBtn = st.form_submit_button(label="Submit")

        if "journalBtn_state" not in st.session_state:
            st.session_state["journalBtn_state"] = False

        if submitJournalBtn or st.session_state["journalBtn_state"]:
            st.session_state["journalBtn_state"] = True

            # Make prediction
            prediction = predict_samples([rawText], BERTmodel)
            keywords = extract_keywords(preprocess_corpus(rawText))
            st.subheader("Based on your journal entry...")

            predictionCol, selectionCol = st.columns(2)
            with predictionCol:
                # Display emotions and their probabilities
                st.success(
                    "These are the top three emotions detected along with their probabilities")
                st.write(prediction)

            with selectionCol:
                # Show form to select emotions and topics
                with st.form(key="selectEmotionsForm"):
                    st.write(
                        "Please select the emotion(s) that best fit your entry for today:")

                    emotion1 = st.checkbox(prediction.iloc[0, 0])
                    emotion2 = st.checkbox(prediction.iloc[1, 0])
                    emotion3 = st.checkbox(prediction.iloc[2, 0])

                    submitCheckboxBtn = st.form_submit_button(
                        label="Submit Selection")

            if submitCheckboxBtn:
                st.subheader("Here is some advice based on...")
                selected_emotions = get_selected_emotions(
                    emotion1, emotion2, emotion3, prediction)
                st.caption("these emotion(s): {}".format(
                    ", ".join(selected_emotions)))

                # Replace the following line with the actual ChatGPT function to generate advice
                st.write(chatgpt.askAdvice(selected_emotions, keywords))

    elif choice == "Details":
        st.subheader("Go Emotion Taxonomy")


if __name__ == '__main__':
    main()
