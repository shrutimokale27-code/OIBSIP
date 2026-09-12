import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset
df = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# 2. Data Cleaning
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()
df = df.dropna()

# 3. Convert Labels
df["label_num"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# 4. Separate Features and Target
X = df["message"]
y = df["label_num"]

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 7. Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 8. Prediction
y_pred = model.predict(X_test_tfidf)

# 9. Model Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"]
))

# 10. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Spam Detection Confusion Matrix")
plt.tight_layout()
plt.show()

# 11. Test Custom Messages
messages = [
    "Congratulations! You have won a free prize. Click now!",
    "Hi, are we meeting for the project today?"
]

message_tfidf = vectorizer.transform(messages)
predictions = model.predict(message_tfidf)

print("\nCustom Message Predictions:")

for message, prediction in zip(messages, predictions):
    result = "Spam" if prediction == 1 else "Ham"
    print(f"{message} -> {result}")