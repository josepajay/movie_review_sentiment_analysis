import numpy as np
import nltk
import random
import sklearn
import operator
import requests
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

# Function to retrieve vocabulary
def get_vocabulary(dataset_file_pos, dataset_file_neg, num_features):
  # stopwords list from nltk
  stopwords=set(nltk.corpus.stopwords.words('english'))
  # words to the stopword list
  stopwords.add(".")
  stopwords.add(",")
  stopwords.add("--")
  stopwords.add("``")
  dict_word_frequency={}
  for pos_review in dataset_file_pos:
    sentence_tokens=get_list_tokens(pos_review)
    for word in sentence_tokens:
      if word in stopwords: continue
      if word not in dict_word_frequency: dict_word_frequency[word]=1
      else: dict_word_frequency[word]+=1
  for neg_review in dataset_file_neg:
    sentence_tokens=get_list_tokens(neg_review)
    for word in sentence_tokens:
      if word in stopwords: continue
      if word not in dict_word_frequency: dict_word_frequency[word]=1
      else: dict_word_frequency[word]+=1
  sorted_list = sorted(dict_word_frequency.items(), key=operator.itemgetter(1), reverse=True)[:num_features]
  vocabulary=[]
  for word,frequency in sorted_list:
    vocabulary.append(word)
  return vocabulary

def get_list_tokens(string): # Function to retrieve the list of tokens from a string
  lemmatizer = nltk.stem.WordNetLemmatizer()
  sentence_split=nltk.tokenize.sent_tokenize(string)
  list_tokens=[]
  for sentence in sentence_split:
    list_tokens_sentence=nltk.tokenize.word_tokenize(sentence)
    for token in list_tokens_sentence:
      list_tokens.append(lemmatizer.lemmatize(token).lower())
  return list_tokens

def get_vector_text(list_vocab,string):
  vector_text=np.zeros(len(list_vocab))
  list_tokens_string=get_list_tokens(string)
  for i, word in enumerate(list_vocab):
    if word in list_tokens_string:
      vector_text[i]=list_tokens_string.count(word)
  return vector_text

def train_svm_classifier(positive_dataset_train, negative_dataset_train, vocabulary): # Function for training svm classifier
  X_train=[]
  Y_train=[]
  for pos_review in positive_dataset_train:
    vector_pos_review=get_vector_text(vocabulary,pos_review)
    X_train.append(vector_pos_review)
    Y_train.append(1)
  for neg_review in negative_dataset_train:
    vector_neg_review=get_vector_text(vocabulary,neg_review)
    X_train.append(vector_neg_review)
    Y_train.append(0)
  # Train the svm binary classifier
  X_train_sentanalysis=np.asarray(X_train)
  Y_train_sentanalysis=np.asarray(Y_train)

  # CHI-SQUARED TEST METHOD
  # To remove features that appears to be irrelevant
  fs_sentanalysis=SelectKBest(chi2, k=500).fit(X_train_sentanalysis, Y_train_sentanalysis)
  X_train_sentanalysis_new = fs_sentanalysis.transform(X_train_sentanalysis)

  svm_clf_sentanalysis=sklearn.svm.SVC(gamma='auto')
  svm_clf_sentanalysis.fit(X_train_sentanalysis_new, Y_train_sentanalysis)
  return svm_clf_sentanalysis, fs_sentanalysis

def main():
    # all the code goes here other than helping functions
    print("Program start, raining model..")
    path='datasets_coursework1/IMDb/train/imdb_train_pos.txt'
    positive_dataset_train=open(path).readlines()
    path='datasets_coursework1/IMDb/train/imdb_train_neg.txt'
    negative_dataset_train=open(path).readlines()

    # invoke get_vocabulary function defined above to obtain vocabulary array
    vocabulary = get_vocabulary(positive_dataset_train, negative_dataset_train, 1000)
    print("\n Model training in progress..")

    svm_clf_sentanalysis, fs_sentanalysis = train_svm_classifier(positive_dataset_train, negative_dataset_train, vocabulary)

    print("\n Model training complete")
    print("\n Predicting for test dataset")
    # Validation of test set data
    path='datasets_coursework1/IMDb/test/imdb_test_pos.txt'
    positive_dataset_test=open(path).readlines()
    path='datasets_coursework1/IMDb/test/imdb_test_neg.txt'
    negative_dataset_test=open(path).readlines()

    # Prepare TEST data set
    X_test=[]
    Y_test=[]
    for instance in positive_dataset_test:
        vector_instance=get_vector_text(vocabulary,instance)
        X_test.append(vector_instance)
        Y_test.append(1)
    for instance in negative_dataset_test:
        vector_instance=get_vector_text(vocabulary,instance)
        X_test.append(vector_instance)
        Y_test.append(0)
    X_test=np.asarray(X_test)
    Y_test_gold=np.asarray(Y_test)


    Y_text_predictions=svm_clf_sentanalysis.predict(fs_sentanalysis.transform(X_test))
    print(classification_report(Y_test_gold, Y_text_predictions))
    accuracy = accuracy_score(Y_test_gold, Y_text_predictions)
    precision = precision_score(Y_test_gold, Y_text_predictions, average='macro')
    f1 = f1_score(Y_test_gold, Y_text_predictions, average='macro')
    recall = recall_score(Y_test_gold, Y_text_predictions, average='macro')

    print ("\n Accuracy : " +str(accuracy))
    print ("\n Precision : " +str(precision))
    print ("\n Recall : " +str(recall))
    print ("\n F1 " +str(f1) + "\n")
    print (confusion_matrix(Y_test_gold, Y_text_predictions))

if __name__== "__main__":
  main()
