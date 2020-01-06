import pandas as pd
df = pd.read_csv("./movie_data.csv")
print(df.loc[0, 'review'][-50:])

import re
def preprocessor(text):
  text = re.sub('<[^>]*>', '', text)
  emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
  text = re.sub('[/W]+', ' ', text.lower()) + ''.join(emoticons).replace('-', '')
  return text


print(preprocessor(df.loc[0, 'review'][-50:]))

#df['review'] = df['review'].apply(preprocessor)

def tokenizer(text):
  return text.split()

print(tokenizer('return hoge hoge'))


from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
# map
def tokenizer_porter(text):
  return [porter.stem(word) for word in text.split()]

print(tokenizer_porter('runners like running and the they making'))


import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = stopwords.words('english')
print([w for w in tokenizer_porter('a runnner likes runnning and runs a lot')[-10:] if w not in stop])

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [{'vect__ngram_range': [(1, 1)], 
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)], 
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf': [False],
               'vect__norm': [None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]}]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
gs_lr_tfidf.fit(X_train,  y_train)


print('Best parametr set: %s' % gs_lr_tfidf.best_params_)