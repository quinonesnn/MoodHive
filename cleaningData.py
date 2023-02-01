# Loading and preparing the models for machine learning models
import pandas as pd


# load the datasets
# csv1 = pd.read_csv(
#     '/Users/nickq/Repos/MoodHive/data/google-research:goemotions/goemotions_1.csv',
#     chunksize=1000)
# data = pd.concat([chunk for chunk in csv1], ignore_index=True)
# csv2 = pd.read_csv(
#     '/Users/nickq/Repos/MoodHive/data/google-research:goemotions/goemotions_2.csv',
#     chunksize=1000)
# data = pd.concat([data] + [chunk for chunk in csv2], ignore_index=True)
# csv3 = pd.read_csv(
#     '/Users/nickq/Repos/MoodHive/data/google-research:goemotions/goemotions_3.csv',
#     chunksize=1000)
# data = pd.concat([data] + [chunk for chunk in csv3], ignore_index=True)


csv1 = pd.read_csv(
    '/Users/nickq/Repos/MoodHive/data/google-research:goemotions/goemotions_1.csv',
    encoding='utf-8')
csv2 = pd.read_csv(
    '/Users/nickq/Repos/MoodHive/data/google-research:goemotions/goemotions_2.csv',
    encoding='utf-8')
csv3 = pd.read_csv(
    '/Users/nickq/Repos/MoodHive/data/google-research:goemotions/goemotions_3.csv',
    encoding='utf-8')

csv1.reset_index(drop=True)
csv2.reset_index(drop=True)
csv3.reset_index(drop=True)

print("null values")
print(csv3.isnull().sum())
print("_____")


data = pd.concat([csv1, csv3], ignore_index=True)
#data = pd.concat([csv2, csv3]).reset_index(drop=True)


print(csv1.shape[0])
print(csv2.shape[0])
print(csv3.shape[0])
print("total: {}".format((csv1.shape[0] + csv2.shape[0] + csv3.shape[0])))

# Combine datasets


# display the first few rows of the dataset
print(csv3.shape[0])

# drop any missing values
data.dropna(inplace=True)

# create a emotion label column
data['emotion'] = ''

# assign the emotion label based on the boolean fields
data.loc[data['admiration'] == 1, 'emotion'] = 'admiration'
data.loc[data['amusement'] == 1, 'emotion'] = 'amusement'
data.loc[data['anger'] == 1, 'emotion'] = 'anger'
data.loc[data['annoyance'] == 1, 'emotion'] = 'annoyance'
data.loc[data['approval'] == 1, 'emotion'] = 'approval'
data.loc[data['caring'] == 1, 'emotion'] = 'caring'
data.loc[data['confusion'] == 1, 'emotion'] = 'confusion'
data.loc[data['curiosity'] == 1, 'emotion'] = 'curiosity'
data.loc[data['desire'] == 1, 'emotion'] = 'desire'
data.loc[data['disappointment'] == 1, 'emotion'] = 'disappointment'
data.loc[data['disapproval'] == 1, 'emotion'] = 'disapproval'
data.loc[data['disgust'] == 1, 'emotion'] = 'disgust'
data.loc[data['embarrassment'] == 1, 'emotion'] = 'embarrassment'
data.loc[data['excitement'] == 1, 'emotion'] = 'excitement'
data.loc[data['fear'] == 1, 'emotion'] = 'fear'
data.loc[data['gratitude'] == 1, 'emotion'] = 'gratitude'
data.loc[data['grief'] == 1, 'emotion'] = 'grief'
data.loc[data['joy'] == 1, 'emotion'] = 'joy'
data.loc[data['love'] == 1, 'emotion'] = 'love'
data.loc[data['nervousness'] == 1, 'emotion'] = 'nervousness'
data.loc[data['optimism'] == 1, 'emotion'] = 'optimism'
data.loc[data['pride'] == 1, 'emotion'] = 'pride'
data.loc[data['realization'] == 1, 'emotion'] = 'realization'
data.loc[data['relief'] == 1, 'emotion'] = 'relief'
data.loc[data['remorse'] == 1, 'emotion'] = 'remorse'
data.loc[data['sadness'] == 1, 'emotion'] = 'sadness'
data.loc[data['surprise'] == 1, 'emotion'] = 'surprise'
data.loc[data['neutral'] == 1, 'emotion'] = 'neutral'


# Cleaning the data and exporting it
cleaned_data = data[['text', 'emotion']]
print(cleaned_data.shape[0])
cleaned_data.to_csv('fullCleanData.csv', index=False)

print("Data saved")
