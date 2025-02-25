import pandas as pd
import numpy as np

import emoji
import contractions
import re

from transformers import TFBertModel, BertTokenizerFast, BertConfig
from keras.models import model_from_json
from keras.optimizers import Adam
from keras import backend as K


# Importing calculated class weights (from the BERT model notebook)
class_weights = np.load(
    '/Users/nickq/Repos/MoodHive/weights/class_wieghts.npy')


# Custom loss function for multilabel
def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:, 0]**(1-y_true))*(weights[:, 1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss


# Loading the model
with open("/Users/nickq/Repos/MoodHive/Models/bert_model_architecture.json", "r") as json_file:
    model_json = json_file.read()


BERTmodel = model_from_json(model_json)
BERTmodel.load_weights(
    "/Users/nickq/Repos/MoodHive/Models/bert-weights.hdf5")

# Compile the model with the custom loss function
optimizer = Adam(learning_rate=2e-5)
loss = get_weighted_loss(class_weights)
BERTmodel.compile(optimizer=optimizer, loss=loss)

# Loading emotion labels for GoEmotions taxonomy
with open("/Users/nickq/Repos/MoodHive/Data/archive/data/emotions.txt", "r") as file:
    GE_taxonomy = file.read().split("\n")
GE_taxonomy.remove('neutral')


# Loading emotion labels for Ekman taxonomy
with open("/Users/nickq/Repos/MoodHive/Data/archive/data/ekman_labels.txt", "r") as file:
    Ekman_taxonomy = file.read().split("\n")
Ekman_taxonomy.remove('neutral')

# Importing BERT pre-trained model and tokenizer
model_name = 'bert-base-uncased'
config = BertConfig.from_pretrained(model_name, output_hidden_states=False)
tokenizer = BertTokenizerFast.from_pretrained(
    pretrained_model_name_or_path=model_name, config=config)
transformer_model = TFBertModel.from_pretrained(model_name, config=config)

# Retrieving initial preprocessings


def preprocess_corpus(x):

    # Adding a space between words and punctation
    x = re.sub(r'([a-zA-Z\[\]])([,;.!?])', r'\1 \2', x)
    x = re.sub(r'([,;.!?])([a-zA-Z\[\]])', r'\1 \2', x)

    # Demojize
    x = emoji.demojize(x)

    # Expand contraction
    x = contractions.fix(x)

    # Lower
    x = x.lower()

    # correct some acronyms/typos/abbreviations
    x = re.sub(r"lmao", "laughing my ass off", x)
    x = re.sub(r"amirite", "am i right", x)
    x = re.sub(r"\b(tho)\b", "though", x)
    x = re.sub(r"\b(ikr)\b", "i know right", x)
    x = re.sub(r"\b(ya|u)\b", "you", x)
    x = re.sub(r"\b(eu)\b", "europe", x)
    x = re.sub(r"\b(da)\b", "the", x)
    x = re.sub(r"\b(dat)\b", "that", x)
    x = re.sub(r"\b(dats)\b", "that is", x)
    x = re.sub(r"\b(cuz)\b", "because", x)
    x = re.sub(r"\b(fkn)\b", "fucking", x)
    x = re.sub(r"\b(tbh)\b", "to be honest", x)
    x = re.sub(r"\b(tbf)\b", "to be fair", x)
    x = re.sub(r"faux pas", "mistake", x)
    x = re.sub(r"\b(btw)\b", "by the way", x)
    x = re.sub(r"\b(bs)\b", "bullshit", x)
    x = re.sub(r"\b(kinda)\b", "kind of", x)
    x = re.sub(r"\b(bruh)\b", "bro", x)
    x = re.sub(r"\b(w/e)\b", "whatever", x)
    x = re.sub(r"\b(w/)\b", "with", x)
    x = re.sub(r"\b(w/o)\b", "without", x)
    x = re.sub(r"\b(doj)\b", "department of justice", x)

    # replace some words with multiple occurences of a letter, example "coooool" turns into --> cool
    x = re.sub(r"\b(j+e{2,}z+e*)\b", "jeez", x)
    x = re.sub(r"\b(co+l+)\b", "cool", x)
    x = re.sub(r"\b(g+o+a+l+)\b", "goal", x)
    x = re.sub(r"\b(s+h+i+t+)\b", "shit", x)
    x = re.sub(r"\b(o+m+g+)\b", "omg", x)
    x = re.sub(r"\b(w+t+f+)\b", "wtf", x)
    x = re.sub(r"\b(w+h+a+t+)\b", "what", x)
    x = re.sub(r"\b(y+e+y+|y+a+y+|y+e+a+h+)\b", "yeah", x)
    x = re.sub(r"\b(w+o+w+)\b", "wow", x)
    x = re.sub(r"\b(w+h+y+)\b", "why", x)
    x = re.sub(r"\b(s+o+)\b", "so", x)
    x = re.sub(r"\b(f)\b", "fuck", x)
    x = re.sub(r"\b(w+h+o+p+s+)\b", "whoops", x)
    x = re.sub(r"\b(ofc)\b", "of course", x)
    x = re.sub(r"\b(the us)\b", "usa", x)
    x = re.sub(r"\b(gf)\b", "girlfriend", x)
    x = re.sub(r"\b(hr)\b", "human ressources", x)
    x = re.sub(r"\b(mh)\b", "mental health", x)
    x = re.sub(r"\b(idk)\b", "i do not know", x)
    x = re.sub(r"\b(gotcha)\b", "i got you", x)
    x = re.sub(r"\b(y+e+p+)\b", "yes", x)
    x = re.sub(r"\b(a*ha+h[ha]*|a*ha +h[ha]*)\b", "haha", x)
    x = re.sub(r"\b(o?l+o+l+[ol]*)\b", "lol", x)
    x = re.sub(r"\b(o*ho+h[ho]*|o*ho +h[ho]*)\b", "ohoh", x)
    x = re.sub(r"\b(o+h+)\b", "oh", x)
    x = re.sub(r"\b(a+h+)\b", "ah", x)
    x = re.sub(r"\b(u+h+)\b", "uh", x)

    # Handling emojis
    x = re.sub(r"<3", " love ", x)
    x = re.sub(r"xd", " smiling_face_with_open_mouth_and_tightly_closed_eyes ", x)
    x = re.sub(r":\)", " smiling_face ", x)
    x = re.sub(r"^_^", " smiling_face ", x)
    x = re.sub(r"\*_\*", " star_struck ", x)
    x = re.sub(r":\(", " frowning_face ", x)
    x = re.sub(r":\^\(", " frowning_face ", x)
    x = re.sub(r";\(", " frowning_face ", x)
    x = re.sub(r":\/",  " confused_face", x)
    x = re.sub(r";\)",  " wink", x)
    x = re.sub(r">__<",  " unamused ", x)
    x = re.sub(r"\b([xo]+x*)\b", " xoxo ", x)
    x = re.sub(r"\b(n+a+h+)\b", "no", x)

    # Handling special cases of text
    x = re.sub(r"h a m b e r d e r s", "hamberders", x)
    x = re.sub(r"b e n", "ben", x)
    x = re.sub(r"s a t i r e", "satire", x)
    x = re.sub(r"y i k e s", "yikes", x)
    x = re.sub(r"s p o i l e r", "spoiler", x)
    x = re.sub(r"thankyou", "thank you", x)
    x = re.sub(r"a^r^o^o^o^o^o^o^o^n^d", "around", x)

    # Remove special characters and numbers replace by space + remove double space
    x = re.sub(r"\b([.]{3,})", " dots ", x)
    x = re.sub(r"[^A-Za-z!?_]+", " ", x)
    x = re.sub(r"\b([s])\b *", "", x)
    x = re.sub(r" +", " ", x)
    x = x.strip()

    return x

# from probabilities to labels using a given threshold


def proba_to_labels(y_pred_proba, top_n=3):
    y_pred_labels = []

    for i in range(y_pred_proba.shape[0]):
        top_indices = np.argsort(y_pred_proba[i])[-top_n:]
        top_probabilities = y_pred_proba[i][top_indices]
        y_pred_labels.append(list(zip(top_indices, top_probabilities)))

    return y_pred_labels


def predict_samples(text_samples, model):

    # Text preprocessing and cleaning
    text_samples_clean = [preprocess_corpus(text) for text in text_samples]

    # Tokenizing train data
    samples_token = tokenizer(
        text=text_samples_clean,
        add_special_tokens=True,
        max_length=48,
        truncation=True,
        padding='max_length',
        return_tensors='tf',
        return_token_type_ids=True,
        return_attention_mask=True,
        verbose=True,
    )

    # Preparing to feed the model
    samples = {'input_ids': samples_token['input_ids'],
               'attention_mask': samples_token['attention_mask'],
               'token_ids': samples_token['token_type_ids']
               }

    # Probability predictions
    samples_pred_proba = model.predict(samples)

    # Label prediction using top N emotions
    samples_pred_labels_with_proba = proba_to_labels(samples_pred_proba)

    # Create a DataFrame with emotions and probabilities
    emotions = []
    probabilities = []

    for sample_labels_with_proba in samples_pred_labels_with_proba:
        for label, proba in sample_labels_with_proba:
            emotions.append(GE_taxonomy[label])
            probabilities.append(proba)

    result_df = pd.DataFrame(
        {"Emotion": emotions, "Probability": probabilities})

    return result_df
