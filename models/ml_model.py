import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
import pickle
warnings.filterwarnings("ignore")

# Load your dataset
df = pd.read_csv(r"C:\Users\Anjel69\Desktop\machinelearning\Symptom2Disease.csv")
df.drop("Unnamed: 0", inplace=True, axis=1)

def clean_text(sent):
    # remove punctuations
    sent = sent.translate(str.maketrans('', '', string.punctuation)).strip()

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(sent)
    words = [word for word in words if word not in stop_words]

    return " ".join(words).lower()

# Apply clean_text on text column of df
df["text"] = df["text"].apply(clean_text)

# Create word cloud to visualize frequent words in our dataset
all_text = " ".join(df["text"])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Use TF-IDF for text vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1500)
tfidf_train = tfidf_vectorizer.fit_transform(X_train).toarray()



# Assuming 'knn' is your trained KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(tfidf_train, y_train)

# Save the model and vectorizer using joblib
joblib.dump(knn, 'classification_model.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')



'''
data = pd.read_csv("C:/Users/Anjel69/Desktop/machinelearning/REngineered_dataset.csv")
data = np.array(data)

X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()


log_reg.fit(X_train, y_train)

inputt=[int(x) for x in "45 32 60".split(' ')]
final=[np.array(inputt)]

b = log_reg.predict_proba(final)


pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
'''
