import openai
import os
import requests

API_KEY = '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'

openai.organization = "org-u2rpGzPE46Zsfqx5Ee2xk0Mg"
openai.api_key = API_KEY

model = 'text-davinci-003'


def askAdvice(emotions, topic="life", keywords=["advice"]):
    # keywords defaults to "advice"
    prompt = "Based on a journal entry, analyze the emotions, themes, and key words that are provided and compile a short list of actionable advice that may help.\n" \
        "The following is the information you should base your next response on:\nEmotions:{}\nTopic:{}\nKeywords:{}\n"\
        "Here's an example of what the response could look like based on the variables you've provided:\n"\
        "Emotion:Sad, Topic:Love, Key Words: Trust, Betrayal, Relationship, Honesty, Forgiveness\n"\
        "\'1. Start by communicating your feelings with your partner. Share with them how you feel and what you need in the relationship.\'\n" \
        "\'2. Remember that forgiveness is a process, and it may take time to fully forgive someone who has betrayed your trust. Take the time you need, but also be willing to work on rebuilding the relationship if that's what you want.\'\n"\
        "Now make your own list with the emotions mentioned earlier".format(
            ", ".join(emotions), topic, ", ".join(keywords))
    response = openai.Completion.create(
        prompt=prompt,
        model=model,
        max_tokens=1000,
        temperature=0
    )
    return response.choices[0].text
