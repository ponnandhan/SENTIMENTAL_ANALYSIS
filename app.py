import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Download stopwords
nltk.download('stopwords')

# 1. Load your CSV dataset
# Make sure to update the filename if it's different
print("Loading dataset from CSV...")
data = pd.read_csv("IMDB Dataset.csv")

# 2. Clean the text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # remove HTML
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove digits
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text

print("Cleaning text data...")
data["review"] = data["review"].apply(clean_text)

# 3. Convert labels to binary if needed
# Supports 'pos/neg' or 1/0
if data["sentiment"].dtype == object:
    data["sentiment"] = data["sentiment"].apply(lambda x: 1 if x.lower() in ["positive", "pos", "1"] else 0)

# 4. Train-Test Split
X = data["review"]
y = data["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. TF-IDF Vectorization
print("Vectorizing with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train the Model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 7. Evaluation
y_pred = model.predict(X_test_vec)
print(f"\nAccuracy: {round(accuracy_score(y_test, y_pred) * 100, 2)}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# 8. Predict new input
def predict_sentiment(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "Positive 😀" if pred == 1 else "Negative 😞"

# 9. Input loop
while True:
    user_input = input("\nEnter a review (or type 'quit' to exit):\n> ")
    if user_input.lower() == "quit":
        break
    print("Predicted Sentiment:", predict_sentiment(user_input))
