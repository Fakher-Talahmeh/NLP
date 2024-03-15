import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
import re

docs = pd.read_json("dataset.json")
docs_text = docs["text"] if "text" in docs else docs.iloc[:, 0]
y = docs.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    docs_text, y, test_size=0.3, random_state=50
)


def clean_data(documents):
    return [re.sub(r"[^\w\s]", "", doc.lower()) for doc in documents]

X_train_clean = clean_data(X_train)
X_test_clean = clean_data(X_test)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train_clean)
X_test_counts = count_vect.transform(X_test_clean)

tfidf_vect = TfidfVectorizer()
X_train_tfidf = tfidf_vect.fit_transform(X_train_clean)
X_test_tfidf = tfidf_vect.transform(X_test_clean)

clf_count = MultinomialNB()
clf_count.fit(X_train_counts, y_train)
y_pred_count = clf_count.predict(X_test_counts)

clf_tfidf = MultinomialNB()
clf_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = clf_tfidf.predict(X_test_tfidf)

accuracy_count = accuracy_score(y_test, y_pred_count)
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)

print(f"Accuracy with CountVectorizer: {accuracy_count}")
print(f"Accuracy with TfidfVectorizer: {accuracy_tfidf}")
plt.figure(figsize=(8, 4))
plt.bar(["Count", "TFIDF"], [accuracy_count, accuracy_tfidf], color=["blue", "green"])
plt.xlabel("Feature Type")
plt.ylabel("Accuracy")
plt.title("Naive Bayes Classification Accuracy")
plt.show()
