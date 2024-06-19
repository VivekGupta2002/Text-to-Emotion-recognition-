import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('./Sentiments/text_emotion.csv')
data.head()
data = data.drop(data.columns[0], axis=1)
sentiment = data.pop('sentiment')
data['sentiment'] = sentiment
data.head()
sentiment = data['sentiment'].value_counts()
sentiment_order = list(sentiment.index)



df = data.copy()
df['content_len'] = df['content'].apply(len)

# table
sentiment_grouped_mean_len = df.groupby('sentiment')['content_len'].mean()
sgmlen_transpose = sentiment_grouped_mean_len.to_frame().transpose()
# type(sentiment_grouped_mean_len)

    
#Data Visualisation

df['content_word'] = data['content'].apply(lambda x: len(x.split()))

# table
sentiment_grouped_mean_word = df.groupby('sentiment')['content_word'].mean()
sgmword_transpose = sentiment_grouped_mean_word.to_frame().transpose()

'''
 print(tabulate(sgmword_transpose.reindex(columns=sentiment_order),
               headers='keys',
               numalign='center',
               stralign='center',
               tablefmt='simple',
               showindex=False))
'''

# barplot
    
#Data Visualisation
 #plt.show()


# Data Visualisation

# # mean word length for each category.

grouped = df.groupby('sentiment')[['content_len', 'content_word']].sum()  # Use double brackets for a list
grouped['avg_word_len'] = grouped['content_len'] / grouped['content_word']

# The number of tweets starting with '@'
count = sum(data['content'].str.startswith('@'))
print(f'The number of replies: {count}')
df_clean = data.copy()

import nltk

from nltk.corpus import stopwords
import re
import string
import contractions
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer


stop_words = set(stopwords.words('english'))

def expand_contractions(text):
    '''
    Function replaces abbreviations with full word versions
    '''
    return contractions.fix(text)

def clean_content(text):
    
    text = expand_contractions(text)
    # remove twitter handles
    clean_text = re.sub(r'@\w+\s?', '', text)
    
    # convert to lowercase
    clean_text = clean_text.lower()
    
    # remove links http:// or https://
    clean_text = re.sub(r'https?:\/\/\S+', '', clean_text)
    
    # remove links beginning with www. and ending with .com
    clean_text = re.sub(r'www\.[a-z]?\.?(com)+|[a-z]+\.(com)', '', clean_text)
    
    # remove html reference characters
    clean_text = re.sub(r'&[a-z]+;', '', clean_text)
    
    # remove non-letter characters besides spaces "/", ";" "[", "]" "=", "#"
    clean_text = re.sub(r"[^a-z\s\(\-:\)\\\/\];='#]", '', clean_text)   
    
    clean_text = clean_text.split()
    
    # remove stop words
    clean_lst = []
    for word in clean_text:
      if word not in stop_words and not word.startswith(('http', '@', 'www.')) and not word.endswith('.com'):
        clean_lst.append(word)
    
    lemmatized_words = []
    for word in clean_lst:
      '''
      # Assign a tag to each part of speech
      tag = pos_tag([word])[0][1][0].upper()
      tag_dict = {'J': wordnet.ADJ,
                  'N': wordnet.NOUN,
                  'V': wordnet.VERB,
                  'R': wordnet.ADV}
      pos = tag_dict.get(tag, wordnet.NOUN)
      
      # lemmatization
      lemmatized_word = WordNetLemmatizer().lemmatize(word, pos)
      '''
      lemmatized_word = WordNetLemmatizer().lemmatize(word)
      lemmatized_words.append(lemmatized_word)

    return ' '.join(lemmatized_words)

df_clean['content'] = df_clean['content'].apply(lambda x :  clean_content(x))

# delete duplicates
df_clean.drop_duplicates(subset='content', inplace=True)
df_clean.reset_index(drop=True, inplace=True)

# delete small sentence
df_clean = df_clean.loc[df_clean['content'].apply(lambda x: len(x) >= 3)]

# splitting into tokens, features of the structure of the text used in Twitter
df_clean['content'] = df_clean['content'].apply(TweetTokenizer().tokenize)

# remove punctuation marks
PUNCTUATION_LIST = list(string.punctuation)
def remove_punctuation(word_list):
    return [w for w in word_list if w not in PUNCTUATION_LIST]
df_clean['content'] = df_clean['content'].apply(remove_punctuation)
df_clean['content'] = df_clean['content'].apply(lambda x: ' '.join(x))


df_clean.info()
df_clean.head()
nltk.download('omw-1.4')
#ntlk.download('wordnet')
nltk.download('wordnet')
from wordcloud import WordCloud

sentiments = df_clean['sentiment'].unique()
sentiments = list(sentiments)
sentiments = list(sentiments) + list(sentiments[:3])

df_train = df_clean.copy()


# Splitting the dataset into training and testing sets

def split_data(df):
    train = df.copy()

    x = np.array(train['content'].values)
    y = np.array(train['sentiment'].values)

    # convert categorical to numeric
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    labels = np.unique(encoder.inverse_transform(y))

    # split data on train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)

    tf = TfidfVectorizer(analyzer='word', max_features=1000, ngram_range=(1,3))
    x_train = tf.fit_transform(x_train).toarray()
    x_test = tf.transform(x_test).toarray()

    return x_train, x_test, y_train, y_test, labels

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def LogR_train(X, x, Y, y, l):
    lr = LogisticRegression()
    lr = LogisticRegression(penalty='l2', C=1)
    lr.fit(X, Y)
    y_pred = lr.predict(x)
    accuracy_test = lr.score(x, y)
    accuracy_train = lr.score(X, Y)
    return accuracy_test, accuracy_train, y, y_pred

X_train, X_test, y_train, y_test, labels = split_data(df_train)

accuracy_test, accuracy_train, y, y_pred = LogR_train(X_train, X_test, y_train, y_test, labels)


print('Accuracy on train: {:.2f}%'.format(accuracy_train*100))
print('Accuracy on test: {:.2f}%'.format(accuracy_test*100))
print('\nClassification Report:\n\n',classification_report(y , y_pred, target_names=[str(l) for l in labels]))

# def split_data(df_train):
#     X = df_train['content']
#     y = df_train['sentiment']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     return X_train, X_test, y_train, y_test, df_train['sentiment'].unique()

# Create stacking with two estimators: Random Forest and Linear SVC, and a logistic regression as the final estimator

def stacking(X, x, Y, y, l):

    estimators = [
      ('rf', RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=10, max_depth = 4, max_features='log2')),
      ('svr', LinearSVC(dual=False, random_state=42))
      ]
    clf_stck = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(penalty='l2', C=1))
    clf_stck.fit(X, Y)

    y_pred = clf_stck.predict(x)

    accuracy_test = clf_stck.score(x, y)
    accuracy_train = clf_stck.score(X, Y)

    return accuracy_test, accuracy_train, y, y_pred

# X_train, X_test, y_train, y_test, labels = split_data(df_train)

# accuracy_test, accuracy_train, y, y_pred = stacking(X_train, X_test, y_train, y_test, labels)

# print('Accuracy on train: {:.2f}%'.format(accuracy_train*100))
# print('Accuracy on test: {:.2f}%'.format(accuracy_test*100))
# print('\nClassification Report:\n\n',classification_report(y , y_pred, target_names=[str(l) for l in labels]))
# conf_matrix(y, y_pred, labels, 'Confusion Matrix Stack')

df_reduce = df_train.copy()

df_reduce['sentiment'] = df_reduce['sentiment'].replace(['happiness', 'enthusiasm', 'surprise'], 'fun')
df_reduce['sentiment'] = df_reduce['sentiment'].replace('boredom', 'sadness')
df_reduce['sentiment'] = df_reduce['sentiment'].replace('hate', 'anger')
df_reduce['sentiment'] = df_reduce['sentiment'].replace(['relief', 'empty'], 'neutral')



X_train, X_test, y_train, y_test, labels = split_data(df_reduce)

# accuracy_test, accuracy_train, y, y_pred = stacking(X_train, X_test, y_train, y_test, labels)
accuracy_test, accuracy_train, y, y_pred = LogR_train(X_train, X_test, y_train, y_test, labels)

print('Accuracy on train: {:.2f}%'.format(accuracy_train*100))
print('Accuracy on test: {:.2f}%'.format(accuracy_test*100))
print('\nClassification Report:\n\n',classification_report(y , y_pred, target_names=[str(l) for l in labels]))

def LogR_train1(X, Y):
    lr = LogisticRegression()
    lr = LogisticRegression(penalty='l2', C=1)
    lr.fit(X, Y)
    return lr

x = data['content'].to_numpy()
y = data['sentiment'].to_numpy()

tf = TfidfVectorizer(analyzer='word', max_features=1000, ngram_range=(1,3))
x = tf.fit_transform(x).toarray()

# convert categorical to numeric
encoder = LabelEncoder()
y = encoder.fit_transform(y)
labels = np.unique(encoder.inverse_transform(y))

lr = LogR_train1(x,y)


# encoder.inverse_transform(y)

def test(text):
    Test = [text]
    Test = tf.transform(Test).toarray()
    ans = lr.predict(Test)
    fin_ans = (encoder.inverse_transform(ans))
    return fin_ans
