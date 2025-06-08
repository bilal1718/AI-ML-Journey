import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def shuffle_data(data):
    return data.sample(frac=1).reset_index(drop=True)

def train_test_split(data, test_size):
    split_point = int(len(data) * (1 - test_size))
    train_data = data[:split_point]
    test_data = data[split_point:]
    return train_data, test_data

def tfidf_from_texts(texts):
    tokenized_docs = [text.lower().split() for text in texts]

    vocab = sorted(set(word for doc in tokenized_docs for word in doc))
    word_index = {word: i for i, word in enumerate(vocab)}
    
    N = len(tokenized_docs)
    V = len(vocab)

    tf_matrix = np.zeros((N, V))
    for i, doc in enumerate(tokenized_docs):
        for word in doc:
            if word in word_index:
                tf_matrix[i][word_index[word]] += 1
        tf_matrix[i] /= len(doc)  

    idf = np.zeros(V)
    for i, word in enumerate(vocab):
        df = sum(1 for doc in tokenized_docs if word in doc)
        idf[i] = np.log((N + 1) / (df + 1)) + 1  

    tfidf_matrix = tf_matrix * idf

    return tfidf_matrix, vocab

data = pd.read_csv('GDA_SVM_NB_SR/Naive_Bayes/spam.csv', encoding='latin1')
data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
data['v1'] = data['v1'].map({'spam': 1, 'ham': 0})
data = shuffle_data(data)

train_data, test_data = train_test_split(data, 0.2)

train_x = train_data['v2'].tolist()
train_y = train_data['v1'].values
test_x = test_data['v2'].tolist()
test_y = test_data['v1'].values

tfidf_matrix, vocab = tfidf_from_texts(train_x)

m = len(train_y)
prior_spam = np.sum(train_y == 1) / m
prior_ham = 1 - prior_spam

spam_indices = np.where(train_y == 1)[0]
ham_indices = np.where(train_y == 0)[0]

spam_tfidf = tfidf_matrix[spam_indices]
ham_tfidf = tfidf_matrix[ham_indices]

spam_word_totals = np.sum(spam_tfidf, axis=0)
ham_word_totals = np.sum(ham_tfidf, axis=0)

total_spam_words = np.sum(spam_word_totals)
total_ham_words = np.sum(ham_word_totals)

V = len(vocab)
likelihood_spam = (spam_word_totals + 1) / (total_spam_words + V)
likelihood_ham = (ham_word_totals + 1) / (total_ham_words + V)

word_to_index = {word: i for i, word in enumerate(vocab)}
spam_probs = {word: likelihood_spam[i] for word, i in word_to_index.items()}
ham_probs = {word: likelihood_ham[i] for word, i in word_to_index.items()}

def predict(sentence):
    words = sentence.lower().split()
    log_prob_spam = np.log(prior_spam)
    log_prob_ham = np.log(prior_ham)

    for word in words:
        if word in spam_probs:
            log_prob_spam += np.log(spam_probs[word])
            log_prob_ham += np.log(ham_probs[word])
    
    return 1 if log_prob_spam > log_prob_ham else 0 

train_preds = [predict(text) for text in train_x]
test_preds = [predict(text) for text in test_x]

print("Train Set Metrics:")
print("Accuracy :", accuracy_score(train_y, train_preds))
print("Precision:", precision_score(train_y, train_preds))
print("Recall   :", recall_score(train_y, train_preds))
print("F1 Score :", f1_score(train_y, train_preds))

print("Test Set Metrics:")
print("Accuracy :", accuracy_score(test_y, test_preds))
print("Precision:", precision_score(test_y, test_preds))
print("Recall   :", recall_score(test_y, test_preds))
print("F1 Score :", f1_score(test_y, test_preds))
