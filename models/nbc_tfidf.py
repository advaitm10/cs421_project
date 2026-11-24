import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

train_df = pd.read_csv('../data/eng_train.csv')
test_df = pd.read_csv('../data/eng_test.csv')

x_train = train_df['text']
y_train = train_df['polarization']

x_test = test_df['text']
y_test = test_df['polarization']

clf = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB(alpha = 1.0)
)
clf.fit(x_train, y_train)

print(classification_report(y_test, clf.predict(x_test)))
print(f"Macro Average F1: {f1_score(y_test, clf.predict(x_test), average='macro')}")