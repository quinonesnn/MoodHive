import streamlit as st
import pandas as pd
import numpy as np
import joblib
import chatgpt


LRpipe = joblib.load(
    "/Users/nickq/Repos/MoodHive/Models/EmotionLogisticRegressionModel-02-01-23.pkl")


def predictEmotion(text):
    return LRpipe.predict([text])[0]


def getPredictProbability(text):
    return LRpipe.predict_proba([text])


def getTopEmotions(probability):
    table = pd.DataFrame(probability, columns=LRpipe.classes_)
    top = table.melt(var_name='Emotion', value_name='Probability').nlargest(
        3, 'Probability')
    return top

# def getTopTopics(probability):
#     table = pd.DataFrame(probability, columns=TopicPipe.classes_)
#     top = table.melt(var_name='Emotion', value_name='Probability').nlargest(
#         3, 'Probability')
#     return top


def getSelectedCheckboxes(boxOptions, box1, box2, box3):
    # This gets the predicted emotions/topics that were selected when asked which best fit the journal
    checkedList = []
    if box1:
        checkedList.append(boxOptions[0])
    if box2:
        checkedList.append(boxOptions[1])
    if box3:
        checkedList.append(boxOptions[2])
    print("getSelectedCheckboxes clicked:", checkedList)

    return checkedList


def makeProbabilityTable(probability):
    return pd.DataFrame(probability, columns=LRpipe.classes_)


def main():
    st.title("MoodHive")
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
            emotionProbability = getPredictProbability(rawText)
            #topicProbability = getPredictProbability(rawText)
            st.subheader("Based on your journal entry...")

            # Display top 3 predictions based on highest probability
            emotionPredictionCol, topicPredictionCol = st.columns(2)
            with emotionPredictionCol:
                topEmotions = getTopEmotions(emotionProbability)
                st.success("These are the top three emotions detected")
                st.write(topEmotions)

            with topicPredictionCol:
                #topTopics = getTopTopics(topicProbability)
                st.success("These are the top three topics detected")
                # st.write(topTopics)

            # Show form to select emotions and topics
            with st.form(key="selectEmotionsForm"):
                st.write(
                    "Please select the emotion(s) and topic(s) that best fit your entry for today:")

                predictedEmotions = [topEmotions.iloc[0, 0],
                                     topEmotions.iloc[1, 0],
                                     topEmotions.iloc[2, 0]]
                # predictedTopics = [topTopics.iloc[0, 0],
                #                    topTopics.iloc[1, 0],
                #                    topTopics.iloc[2, 0]]

                emotionSelectionCol, topicSelectionCol = st.columns(2)

                with emotionSelectionCol:
                    emotion1 = st.checkbox(predictedEmotions[0])
                    emotion2 = st.checkbox(predictedEmotions[1])
                    emotion3 = st.checkbox(predictedEmotions[2])
                with topicSelectionCol:
                    topic1 = st.checkbox("topic1")
                    topic2 = st.checkbox("topic2")
                    topic3 = st.checkbox("topic3")
                    # topic1 = st.checkbox(predictedTopics[0])
                    # topic2 = st.checkbox(predictedTopics[1])
                    # topic3 = st.checkbox(predictedTopics[2])

                submitCheckboxBtn = st.form_submit_button(
                    label="Submit Selection")

            if submitCheckboxBtn:
                st.subheader("Here is some advice based on...")
                selectedEmotions = getSelectedCheckboxes(
                    predictedEmotions, emotion1, emotion2, emotion3)
                # selectedTopics = getSelectedCheckboxes(
                #     predictedTopics, topic1, topic2, topic3)
                st.caption("these emotion(s): {}".format(
                    ", ".join(selectedEmotions)))
                # st.caption("these topics(s): {}".format(
                #     ", ".join(selectedTopics)))

                # st.write(chatgpt.askAdvice(predictedEmotions))

    elif choice == "Details":
        st.subheader("Details")


if __name__ == '__main__':
    main()
