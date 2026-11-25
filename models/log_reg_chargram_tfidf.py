import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union

train_df = pd.read_csv('../data/eng_train.csv')
test_df = pd.read_csv('../data/eng_test.csv')

x_train = train_df['text']
y_train = train_df['polarization']

x_test = test_df['text']
y_test = test_df['polarization']

char_word = make_union(
    TfidfVectorizer(ngram_range=(1, 2)),
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
)

clf = make_pipeline(
    char_word,
    LogisticRegression(max_iter = 2000, class_weight = 'balanced')
)
clf.fit(x_train, y_train)

print(classification_report(y_test, clf.predict(x_test)))
print(f"Macro Average F1: {f1_score(y_test, clf.predict(x_test), average='macro')}")

# Dump results to pickle
df = pd.DataFrame({
    'text': x_test,
    'true_label': y_test,
    'pred_label': clf.predict(x_test)
})

df.to_pickle('best_model_full_results.pkl')