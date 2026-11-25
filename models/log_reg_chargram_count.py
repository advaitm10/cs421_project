import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline, make_union

train_df = pd.read_csv('../data/eng_train.csv')
test_df = pd.read_csv('../data/eng_test.csv')

x_train = train_df['text']
y_train = train_df['polarization']

x_test = test_df['text']
y_test = test_df['polarization']

char_word = make_union(
    CountVectorizer(ngram_range=(1, 2), stop_words='english'),
    CountVectorizer(analyzer="char_wb", ngram_range=(2, 5), stop_words='english')
)

clf = make_pipeline(
    char_word,
    LogisticRegression(max_iter = 2000, class_weight = 'balanced')
)
clf.fit(x_train, y_train)

print(classification_report(y_test, clf.predict(x_test)))
print(f"Macro Average F1: {f1_score(y_test, clf.predict(x_test), average='macro')}")