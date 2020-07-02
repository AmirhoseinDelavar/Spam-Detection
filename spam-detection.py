import string

import pandas as pd
from PersianStemmer import PersianStemmer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import pickle

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

Stopwords0 = ['و', 'از', 'در', 'که', 'برای', 'چون', 'اما', 'شاید', 'اگر', 'با', 'یا', "آن", 'اون', 'این', 'ازین',
              'ازون',
              'هم', 'های', 'ها', 'یه', 'دو', 'عدد',
              'سه', 'ام', 'ای', 'اید', 'شد', 'است',
              'بود', 'مثل', 'مثلا', 'من', 'شما', 'ما', 'دیگر', 'علاوه', 'مون', 'تون', 'بر']
df = pd.read_csv('./train.csv')
with open('Stopwords.txt') as f:
    Stopwords1 = [line.strip() for line in f]

Stopwords = Stopwords0.extend(Stopwords1)
print(df.head())

# deleting punctuation
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
          '+', '\\', '•', '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
          '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
          'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹',
          '≤', '‡', '√', '1', '2', '3', '4', '5'
                                             '6', '7', '8', '9', '0', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹',
          '۰', ]

dataframelist = df['comment'].values.tolist()

# ps = PersianStemmer()
for i in range(df.shape[0]):
    print(i)
    x = dataframelist[i]
    for punct in puncts:
        if punct in puncts:
            x = str(x).replace(punct, '', -1)

    for word in Stopwords0:
        while x.__contains__(word):
            x = str(x).replace(word, '', -1)
    dataframelist[i] = x

# X_train, X_test, y_train, y_test = train_test_split(dataframelist,
#                                                     df['id'],
#                                                     random_state=1)
newdf = pd.DataFrame({'newcomment': dataframelist})
count_vector = CountVectorizer(lowercase=False, encoding='utf-8')
training_data = count_vector.fit_transform(newdf)

dftest = pd.read_csv('./test.csv')
testing_data = count_vector.transform(dftest['comment'].values.astype('U'))

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, df['id'].values.astype('U'))

predictions = naive_bayes.predict(testing_data)
print(predictions)
