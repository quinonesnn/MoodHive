import os
from dotenv import load_dotenv
import openai

load_dotenv()


API_KEY = os.getenv("OPENAI_API_KEY")
ORG_ID = os.getenv("OPENAI_ORG_ID")

openai.organization = ORG_ID
openai.api_key = API_KEY

model = 'text-davinci-003'


def askAdvice(emotions, keywords=["advice"]):
    # keywords defaults to "advice"
    prompt = "Based on the following journal entry information, analyze the emotions, themes, and keywords provided."\
        "Then, create a comprehensive and personalized list of actionable advice that may help address these emotions and themes"\
        "\nEmotions: {}\nKeywords: {}\n"\
        "Now, create your own comprehensive and personalized list of advice based on the emotions and keywords provided, considering the different aspects mentioned above."\
        .format(", ".join(emotions), ", ".join(keywords))
    response = openai.Completion.create(
        prompt=prompt,
        model=model,
        max_tokens=1000,
        temperature=0
    )
    return response.choices[0].text


# button that Randomize the reponse or generate another response

# a way to keep a session so that it considers the previous response
