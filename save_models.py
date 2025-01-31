# save_models.py
import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load the data
data_fake = pd.read_csv("Fake.csv")
data_true = pd.read_csv("True.csv")

# Add the class labels
data_fake["class"] = 0
data_true["class"] = 1

# Remove last 10 entries for manual testing
data_fake_manual_testing = data_fake.tail(10)
for i in range(23480, 23470, -1):
    data_fake.drop([i], axis=0, inplace=True)

data_true_manual_testing = data_true.tail(10)
for i in range(21416, 21406, -1):
    data_true.drop([i], axis=0, inplace=True)

# Merge the data
data_merge = pd.concat([data_fake, data_true], axis=0)

# Drop unnecessary columns
data = data_merge.drop(['title', 'subject', 'date'], axis=1)

# Reset index
data = data.sample(frac=1)
data.reset_index(inplace=True)
data.drop(['index'], axis=1, inplace=True)

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Preprocess the text
data['text'] = data['text'].apply(wordopt)

# Split into features and target
x = data['text']
y = data['class']

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Initialize and fit the vectorizer
print("Training vectorizer...")
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Train the model
print("Training model...")
LR = LogisticRegression()
LR.fit(xv_train, y_train)

# Save the vectorizer
print("Saving vectorizer...")
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorization, file)

# Save the model
print("Saving model...")
with open('model.pkl', 'wb') as file:
    pickle.dump(LR, file)

# Print accuracy
accuracy = LR.score(xv_test, y_test)
print(f"\nModel accuracy: {accuracy*100:.2f}%")
print("\nFiles saved successfully: 'vectorizer.pkl' and 'model.pkl'")