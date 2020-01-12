import numpy as np
import nltk
import random
import sklearn
import operator
import requests
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score
nltk.download('stopwords') # If needed
nltk.download('punkt') # If needed
nltk.download('wordnet') # If needed
lemmatizer = nltk.stem.WordNetLemmatizer()
stopwords=set(nltk.corpus.stopwords.words('english'))
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")


def train_svm_classifier(training_set, vocabulary): # Function for training our svm classifier
  X_train=[]
  Y_train=[]
  for instance in training_set:
    vector_instance=get_vector_text(vocabulary,instance[0])
    X_train.append(vector_instance)
    Y_train.append(instance[1])
  # Finally, we train the SVM classifier
  svm_clf=sklearn.svm.SVC(kernel="linear",gamma='auto')
  svm_clf.fit(np.asarray(X_train),np.asarray(Y_train))
  return svm_clf

def get_vocabulary(training_set, num_features): # Function to retrieve vocabulary
  dict_word_frequency={}
  for instance in training_set:
    sentence_tokens=get_list_tokens(instance[0])
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

path='datasets_coursework1/Hateval/hateval.tsv'
processed_dataset = []
dataset_file=open(path).readlines()
# Remove heading row
dataset_file.pop(0)
for tweet in dataset_file:
    temp_tweet_holder = tweet.rstrip("\n").split("\t")
    processed_dataset.append((temp_tweet_holder[1], int(temp_tweet_holder[2])))
print (processed_dataset[0])


kf = KFold(n_splits=10)
random.shuffle(processed_dataset)
kf.get_n_splits(processed_dataset)
for train_index, test_index in kf.split(processed_dataset):
  train_set_fold=[]
  test_set_fold=[]
  accuracy_total=0.0
  for i,instance in enumerate(processed_dataset):
    if i in train_index:
      train_set_fold.append(instance)
    else:
      test_set_fold.append(instance)
  vocabulary_fold=get_vocabulary(train_set_fold, 500)
  svm_clf_fold=train_svm_classifier(train_set_fold, vocabulary_fold)
  X_test_fold=[]
  Y_test_fold=[]
  for instance in test_set_fold:
    vector_instance=get_vector_text(vocabulary_fold,instance[0])
    X_test_fold.append(vector_instance)
    Y_test_fold.append(instance[1])
  Y_test_fold_gold=np.asarray(Y_test_fold)
  X_test_fold=np.asarray(X_test_fold)
  Y_test_predictions_fold=svm_clf_fold.predict(X_test_fold)
  accuracy_fold=accuracy_score(Y_test_fold_gold, Y_test_predictions_fold)
  precision_fold = precision_score(Y_test_fold_gold, Y_test_predictions_fold, average='macro')
  f1_fold = f1_score(Y_test_fold_gold, Y_test_predictions_fold, average='macro')
  recall_fold = recall_score(Y_test_fold_gold, Y_test_predictions_fold, average='macro')
  print ("\n Accuracy : " +str(accuracy_fold))
  print ("\n Precision : " +str(precision_fold))
  print ("\n Recall : " +str(recall_fold))
  print ("\n F1 " +str(f1_fold))
  accuracy_total+=accuracy_fold
  print ("Fold completed.")
average_accuracy=accuracy_total/10
print ("\nAverage Accuracy: "+str(round(accuracy_fold,3)))
