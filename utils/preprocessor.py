from utils import config
import nltk
import re
import torch
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os


class YelpDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

def clean_text(text, stop_words):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = text.split(" ")
    text = [word for word in text if not word in stop_words]
    # text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    # text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = " ".join(text)
    return text

def categorize_stars(stars):
    if stars > 3:
        return 'POSITIVE'
    elif stars <= 2:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

def preprocessData(yelp_data, test_size = 0.1, balance = False):
    custom_nltk_data_path = 'nltk/'

    # Add the custom path to NLTK data path
    nltk.data.path.append(custom_nltk_data_path)

    # Check if 'stopwords' are downloaded, and if not, download them
    if not os.path.isfile(os.path.join(custom_nltk_data_path, 'corpora/stopwords.zip')):
        nltk.download('stopwords', download_dir=custom_nltk_data_path)

    # Check if 'wordnet' is downloaded, and if not, download it
    if not os.path.isfile(os.path.join(custom_nltk_data_path, 'corpora/wordnet.zip')):
        nltk.download('wordnet', download_dir=custom_nltk_data_path)
    stop_words = set(stopwords.words("english"))
    # lemmatizer = WordNetLemmatizer()

    yelp_data['Processed_Reviews'] = yelp_data.text.apply(lambda x: clean_text(x, stop_words))

    # Apply star categorization to 'stars' column
    yelp_data['label'] = yelp_data['stars'].apply(categorize_stars)

    category_one_hot = pd.get_dummies(yelp_data['label'])

	# Concatenate the one-hot encoded columns with the original DataFrame
    yelp_data = pd.concat([yelp_data, category_one_hot], axis=1)

    if balance:
        positive_data = yelp_data[yelp_data['label'] == 'POSITIVE'].sample(n=20000, replace=True)
        negative_data = yelp_data[yelp_data['label'] == 'NEGATIVE'].sample(n=20000, replace=True)
        neutral_data = yelp_data[yelp_data['label'] == 'NEUTRAL']

        # Concatenate the balanced subsets
        yelp_data = pd.concat([positive_data, negative_data, neutral_data])

    if test_size == 0:
        return yelp_data['text'].values, None, yelp_data[['POSITIVE', 'NEUTRAL', 'NEGATIVE']].values, None

	# Split dataset into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
	    yelp_data['text'].values,
	    yelp_data[['POSITIVE', 'NEUTRAL', 'NEGATIVE']].values,
	    test_size=test_size,
	    random_state=42
	)

    return train_texts, val_texts, train_labels, val_labels