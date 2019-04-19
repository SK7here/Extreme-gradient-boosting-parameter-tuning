#LOADING LIBRARIES

import re    # for regular expressions 
import nltk  # for text manipulation 
import string # forString operations (not necessary)
import warnings # for throwing exceptions
import numpy as np # for scientific computing
import pandas as pd # for working with data
import seaborn as sns # extension of matplotlib
import matplotlib.pyplot as plt #plots graphs

from nltk.stem.porter import * #To use PorterStemmer function
from wordcloud import WordCloud #To use wordcloud

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer #TfidfVectorizer->gives higher weight to words occuring rare in a entire corpus but good in few documents
import gensim #Topic modelling(identify which topic is discussed), Similarity retrieval

from tqdm import tqdm
tqdm.pandas(desc="progress-bar") #To display progress bar with title "progres-bar" 
from gensim.models.deprecated.doc2vec import LabeledSentence #Labelling tweets for doc2vec purpose

from sklearn.linear_model import LogisticRegression # To build Logistic regression model
from sklearn.model_selection import train_test_split # For splitting data into train and test data
from sklearn.metrics import f1_score # To compute performance of the model
from sklearn import svm # To build Support vector machine model
from sklearn.ensemble import RandomForestClassifier #To build Random forest classifier model
from xgboost import XGBClassifier #To build extreme gradient boosting model
import xgboost as xgb #Imports all features of extreme gradient boosting algorithm

###################################################################################################################################################################

#DATA INSPECTION

#sets value
pd.set_option("display.max_colwidth", 200) 
#To ignore deprecation  warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


#importing datasets
train  = pd.read_csv('train_tweets.csv') 
test = pd.read_csv('test_tweets.csv')


#first 10 data of non-racist and racist tweets resp..,
print("\nNON RACIST TWEETS\n")
print(train[train['label'] == 0].head(10))
print("\nRACIST TWEETS\n")
print(train[train['label'] == 1].head(10))


#dimensions of data set
print("\nTRAINING SET\n")
print(train.shape)
print("\nTEST SET\n")
print(test.shape)


#split of tweets in terms of labels in training set
print("NO OF POSITIVE AND NEGATIVE TWEETS IN TRAINING DATASET")
print(train["label"].value_counts())


#tweets in terms of number of words in each tweet
length_train = train['tweet'].str.len() 
length_test = test['tweet'].str.len()
    #histogram autoamatically scales dataset suitable to plot the graph
    #here bins seperate the enitre dataset into intervals of 20 and plot graph (discrete kind)
plt.hist(length_train, bins=20, label="train_tweets") 
plt.hist(length_test, bins=20, label="test_tweets") 
plt.legend()
plt.xlabel("Tweet id")
plt.ylabel("No of Words")
plt.show()


###################################################################################################################################################################

#DATA CLEANING

#To combine train and test data for cleaning
combi = train.append(test, ignore_index=True , sort=False) 


#user-defined function to remove unwanted text patterns from the tweets.
def remove_pattern(input_txt, pattern):
    #Finds all words matching the pattern
    r = re.findall(pattern, input_txt)
    for i in r:
        #removes the matched words
        input_txt = re.sub(i, '', input_txt)
    return input_txt


#Removing user handles (\w - words)
    #vectorize function used when recursively a function is called
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*") 


#Removing Punctuations, Numbers, and Special Characters (^ - except)
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ") 


#Removing short words (assuming that less than 3 letter words will not much influence over sentiment)
    #lambda function is similar to macros
    #apply funcion applies particular function over every element
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


#Tokenization (List for each tweet where items are each word in the tweet)
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())


#Normalization 
tokenized_tweet = tokenized_tweet.apply(lambda x: [PorterStemmer().stem(i) for i in x])
print("\nFIRST 5 PROCESSED TOKENIZED TWEETS\n")
print(tokenized_tweet.head())

#Stitching normalized tokens together
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])    
combi['tidy_tweet'] = tokenized_tweet


###################################################################################################################################################################


#VISUALIZATION FROM TWEETS

#word cloud visualization is used to identify frequency of words

#Non-racist tweets
    #Taking non - racist tweets
normal_words =' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])
    #Generating word cloud
wordcloud = WordCloud(width=800, height=500, random_state=11, max_font_size=110).generate(normal_words)
    #Plotting word cloud in graph
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="none")
plt.axis('off')
plt.show()


#Racist tweets
    #Taking non - racist tweets
normal_words =' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])
    #Generating word cloud
wordcloud = WordCloud(width=800, height=500, random_state=11, max_font_size=110).generate(normal_words)
    #Plotting word cloud in graph
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="none")
plt.axis('off')
plt.show()


#Function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        #r - raw string used for specifying regular expressions
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags

# extracting hashtags from non-racist/sexist tweets 
HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0])


# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])


# unnesting list (to make muliple list as single list)
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])


#Plotting non-racist hashtags
    #Frequency of each item
a = nltk.FreqDist(HT_regular)
    #Storing frequency as dict/2D form
d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
    # selecting top 20 most frequent hashtags
d = d.nlargest(columns="Count", n = 20)
    #Plotting
plt.figure(figsize=(18,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
plt.show()


#Plotting racist hashtags
    #Frequency of each item
a = nltk.FreqDist(HT_negative)
    #Storing frequency as dict/2D form
d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
    # selecting top 20 most frequent hashtags
d = d.nlargest(columns="Count", n = 20)
    #Plotting
plt.figure(figsize=(18,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
plt.show()



###################################################################################################################################################################

#Bag-of-words Features
    #max_df/min_df = int->no of documents ; float->percentage among total documents
    #max_features = top 1000 features taken
    #stop_words = english-> inbulit stop words list for english is used
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')


#making data to center with mean zero and unit standard deviation
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])



###################################################################################################################################################################

#TF-IDF Features
    #TF = (Number of times term t appears in a document)/(Number of terms in the document)
    #IDF = log(N/n), where, N is the number of documents and n is the number of documents a term t has appeared in.
    #TF-IDF = TF*IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')


#making data to center withe mean zero and unit standard deviation
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])


###################################################################################################################################################################


#_____________________________________________________________Notes on Word2Vec________________________________________________________________________________
    #Word embeddings-> representing words as vectors
    #               -> high dimensional word features into low dimensional feature vectors by preserving the contextual similarity

    
    #Combination of  CBOW (Continuous bag of words) and Skip-gram model.
        #CBOW-> tends to predict the probability of a word given a context
        #Skip-gram model-> tries to predict the context for a given word.


    #Softmax-> converts vector as probability distribution

    
    #Pretrained Word2Vec models (huge in size)
        #Google News Word Vectors
        #Freebase names
        #DBPedia vectors (wiki2vec)
    




#Training own Word2Vec model
    #tokenizing
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())


model_w2v = gensim.models.Word2Vec(
            tokenized_tweet,
            size=200, # to represent in no.of.dimensions-> more the dimension, more the efficiency of model
            window=5, # takes 10 words surrounding the current word to find context of current word 
            min_count=2, # removes words with frequency less than 2
            sg = 1, # 1 for skip-gram model
            hs = 0, # 0 for negative sampling
            negative = 10, # only takes random 10 negative samples(since dataset is huge, this is done)
            workers= 2, # no.of threads to train the model
            seed = 11)  # to generate same random numbers every time


#Training the built model-> should specify model, size of corpus, epoch
model_w2v.train(tokenized_tweet, total_examples= len(combi['tidy_tweet']), epochs=20)


#Getting similar context words to the mentioned word
sim_dinner = model_w2v.wv.most_similar(positive="dinner")
sim_trump = model_w2v.wv.most_similar(positive="trump")


i=0
sim_dinner_len = len(sim_dinner)
print("\nWORDS SIMILAR TO DINNER\n")
while(i<sim_dinner_len):
    print("\n")
    print(sim_dinner[i])
    i=i+1
i=0
print("\nWORDS SIMILAR TO TRUMP\n")
sim_trump_len = len(sim_trump)
while(i<sim_trump_len):
    print("\n")
    print(sim_trump[i])
    i=i+1





#Functions for converting tweets into vectors
def word_vector(tokens, size):
    #Creates array with specified all filled with zeros and it is given a new shape
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            #every token in a tweet is converted as a vector
            vec += model_w2v[word].reshape((1, size))
            count += 1.
        # handling the case where the token is not in vocabulary continue
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


#word2vec feature set
wordvec_arrays = np.zeros((len(tokenized_tweet), 200)) 
for i in range(len(tokenized_tweet)):
    wordvec_arrays[i,:] = word_vector(tokenized_tweet[i], 200)
#converting array into 2D Table
wordvec_df = pd.DataFrame(wordvec_arrays)



###################################################################################################################################################################

#Doc2vec Embedding

#To label each tweet
def add_label(twt):
    output = []
    for i, s in zip(twt.index, twt):
        #Tweet Id's are itself made as label
        output.append(LabeledSentence(s, ["tweet_" + str(i)]))
    return output
labeled_tweets = add_label(tokenized_tweet)
#Prints o to 5 labeled tweets
print("\nFIRST 6 LABELED TWEETS FOR ILLUSTRATION\n")
print(labeled_tweets[:6])


#Doc2vec is a small extension to the CBOW model.
#Instead of using just words to predict the next word, we also added paragraph ID which is document-unique.


#Training doc2vec model
model_d2v = gensim.models.Doc2Vec(
            dm=1, # dm = 1 for ‘distributed memory’ model
            dm_mean=1, # dm = 1 for using mean of the context word vectors
            vector_size=200, # to represent in no.of.dimensions-> more the dimension, more the efficiency of model                                 
            window=5, # takes 10 words surrounding the current word to find context of current word                                   
            negative = 7, # only takes random 7 negative samples(since dataset is huge, this is done)
            min_count=2, # removes words with frequency less than 2                                  
            workers=3, # no. of threads to train the model                                  
            alpha=0.1, # learning rate                                  
            seed = 11) # to generate same random numbers every time


#To show progress bar while training Doc2vec model
model_d2v.build_vocab([i for i in tqdm(labeled_tweets)])
#Training Doc2vec model
model_d2v.train(labeled_tweets, total_examples= len(combi['tidy_tweet']), epochs=15)


#Preparing doc2vec Feature Set
docvec_arrays = np.zeros((len(tokenized_tweet), 200)) 
for i in range(len(combi)):
    docvec_arrays[i,:] = model_d2v.docvecs[i].reshape((1,200))    
#converting array into 2D Table
docvec_df = pd.DataFrame(docvec_arrays) 


###################################################################################################################################################################


#Building different machine learning models to use later

# Creating Logistic regression model
    #solver -> algorithm for optimimzation problem
lreg = LogisticRegression(solver = "lbfgs")

# Creating SVM model
    #kernel -> type of classifier
    #C -> regularization ; lower value of c, larger is the margin seperating hyperplane 
svc = svm.SVC(kernel='linear', C=1, probability=True)

#Creating Random Forest model
    #n_estimators -> No of trees
rf = RandomForestClassifier(n_estimators=400, random_state=11)

#Creating xgb model
    #max_depth -> tree depth ; more the depth, more is the complexity
xgb_cl = XGBClassifier(random_state=11, n_estimators=1000)


#We are going to split the bow, tfidf, word2vec and doc2vec features into train and test data once (in Logitic regression model)
#Use the same train and test data correspondingly in all other machine leaning models also

###################################################################################################################################################################

print("\n\nLOGISTIC REGRESSION MODEL ACCURACIES\n\n")

#BOW FEATURES - Logistic Regression model 
# Segregating dataset into train and test BoW features
    #0 to 31961 -> Training dataset
    #31962 to end -> Test dataset
train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

# splitting Training dataset into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'],random_state=11,test_size=0.3) 
# Training the model with training set from Training dataset
lreg.fit(xtrain_bow, ytrain)
# prediction on the validation set from Training dataset
prediction = lreg.predict_proba(xvalid_bow)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
bow_lreg_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR BAG OF WORDS USING LOGISTIC REGRESSION\n")
print(bow_lreg_pred_score)

#Testing the built Logistic regression model on test data

#predicting on test data
bow_lreg_test_pred = lreg.predict_proba(test_bow)
    #If probability for label 1 is over 0.3, it is taken as label 1
bow_lreg_test_pred_int = bow_lreg_test_pred[:,1]>=0.3
bow_lreg_test_pred_int = bow_lreg_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = bow_lreg_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('logreg_bow.csv', index=False)





#TF-IDF FEATURES - Logistic Regression model
# Segregating dataset into train and test TF-IDF features
    #0 to 31961 -> Training dataset
    #31962 to end -> Test dataset
train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]

# splitting Training dataset into training and validation set (taking same train and validation set of BOW FEATURES)
xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]
# Training the model with training set from Training dataset
lreg.fit(xtrain_tfidf, ytrain)
# prediction on the validation set from Training dataset
prediction = lreg.predict_proba(xvalid_tfidf)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
tfidf_lreg_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR TF-IDF USING LOGISTIC REGRESSION\n")
print(tfidf_lreg_pred_score)

#Testing the built Logistic regression model on test data

#predicting on test data
tfidf_lreg_test_pred = lreg.predict_proba(test_tfidf)
    #If probability for label 1 is over 0.3, it is taken as label 1
tfidf_lreg_test_pred_int = tfidf_lreg_test_pred[:,1]>=0.3
tfidf_lreg_test_pred_int = tfidf_lreg_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = tfidf_lreg_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('logreg_tfidf.csv', index=False)





#WORD2VEC FEATURES - Logistic Regression model
# Segregating dataset into train and test WORD2VEC features
    #0 to 31961 -> Training dataset
    #31962 to end -> Test dataset
    #iloc - access data by row index(since wordvec_df is 2D)
train_w2v = wordvec_df.iloc[:31962,:]
test_w2v = wordvec_df.iloc[31962:,:]

# splitting Training dataset into training and validation set (taking same train and validation set of BOW FEATURES)
xtrain_w2v = train_w2v.iloc[ytrain.index,:]
xvalid_w2v = train_w2v.iloc[yvalid.index,:]
# Training the model with training set from Training dataset
lreg.fit(xtrain_w2v, ytrain)
# prediction on the validation set from Training dataset
prediction = lreg.predict_proba(xvalid_w2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
w2v_lreg_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR WORD2VEC USING LOGISTIC REGRESSION\n")
print(w2v_lreg_pred_score)

#Testing the built Logistic regression model on test data

#predicting on test data
w2v_lreg_test_pred = lreg.predict_proba(test_w2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
w2v_lreg_test_pred_int = w2v_lreg_test_pred[:,1]>=0.3
w2v_lreg_test_pred_int = w2v_lreg_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = w2v_lreg_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('logreg_w2v.csv', index=False)





#DOC2VEC FEATURES - Logistic Regression model
# Segregating dataset into train and test DOC2VEC features
    #0 to 31961 -> Training dataset
    #31962 to end -> Test dataset
    #iloc - access data by row index(since docvec_df is 2D)
train_d2v = docvec_df.iloc[:31962,:]
test_d2v = docvec_df.iloc[31962:,:]

# splitting Training dataset into training and validation set (taking same train and validation set of BOW FEATURES)
xtrain_d2v = train_d2v.iloc[ytrain.index,:]
xvalid_d2v = train_d2v.iloc[yvalid.index,:]
# Training the model with training set from Training dataset
lreg.fit(xtrain_d2v, ytrain)
# prediction on the validation set from Training dataset
prediction = lreg.predict_proba(xvalid_d2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
d2v_lreg_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR DOC2VEC USING LOGISTIC REGRESSION\n")
print(d2v_lreg_pred_score)

#Testing the built Logistic regression model on test data

#predicting on test data
d2v_lreg_test_pred = lreg.predict_proba(test_d2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
d2v_lreg_test_pred_int = d2v_lreg_test_pred[:,1]>=0.3
d2v_lreg_test_pred_int = d2v_lreg_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = d2v_lreg_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('logreg_d2v.csv', index=False)



###################################################################################################################################################################

print("\n\nSUPPORT VECTOR MACHINE MODEL ACCURACIES\n\n")

#BOW FEATURES - SVM model
# Training the model with training set from Training dataset
svc.fit(xtrain_bow, ytrain)
# prediction on the validation set from Training dataset
prediction = svc.predict_proba(xvalid_bow)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
bow_svm_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR BAG OF WORDS USING SUPPORT VECTOR MACHINE\n")
print(bow_svm_pred_score)

#Testing the built SVM model on test data

#predicting on test data
bow_svm_test_pred = svc.predict_proba(test_bow)
    #If probability for label 1 is over 0.3, it is taken as label 1
bow_svm_test_pred_int = bow_svm_test_pred[:,1]>=0.3
bow_svm_test_pred_int = bow_svm_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = bow_svm_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('svm_bow.csv', index=False)





#TF-IDF FEATURES - SVM model
# Training the model with training set from Training dataset
svc.fit(xtrain_tfidf, ytrain)
# prediction on the validation set from Training dataset
prediction = svc.predict_proba(xvalid_tfidf)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
tfidf_svm_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR TF-IDF USING SUPPORT VECTOR MACHINE\n")
print(tfidf_svm_pred_score)

#Testing the built SVM on test data

#predicting on test data
tfidf_svm_test_pred = svc.predict_proba(test_tfidf)
    #If probability for label 1 is over 0.3, it is taken as label 1
tfidf_svm_test_pred_int = tfidf_svm_test_pred[:,1]>=0.3
tfidf_svm_test_pred_int = tfidf_svm_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = tfidf_svm_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('svm_tfidf.csv', index=False)





#WORD2VEC FEATURES - SVM model
# Training the model with training set from Training dataset
svc.fit(xtrain_w2v, ytrain)
# prediction on the validation set from Training dataset
prediction = svc.predict_proba(xvalid_w2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
w2v_svm_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR WORD2VEC USING SUPPORT VECTOR MACHINE\n")
print(w2v_svm_pred_score)

#Testing the built SVM model on test data

#predicting on test data
w2v_svm_test_pred = svc.predict_proba(test_w2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
w2v_svm_test_pred_int = w2v_svm_test_pred[:,1]>=0.3
w2v_svm_test_pred_int = w2v_svm_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = w2v_svm_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('svm_w2v.csv', index=False)





#DOC2VEC FEATURES - SVM model
# Training the model with training set from Training dataset
svc.fit(xtrain_d2v, ytrain)
# prediction on the validation set from Training dataset
prediction = svc.predict_proba(xvalid_d2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
d2v_svm_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR DOC2VEC USING SUPPORT VECTOR MACHINE\n")
print(d2v_svm_pred_score)

#Testing the built SVM model on test data

#predicting on test data
d2v_svm_test_pred = svc.predict_proba(test_d2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
d2v_svm_test_pred_int = d2v_svm_test_pred[:,1]>=0.3
d2v_svm_test_pred_int = d2v_svm_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = d2v_svm_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('svm_d2v.csv', index=False)


###############################################################################################################################################################################

print("\n\nRANDOM FOREST MODEL ACCURACIES\n\n")

#BOW FEATURES - RANDOMFOREST model
# Training the model with training set from Training dataset
rf.fit(xtrain_bow, ytrain)
# prediction on the validation set from Training dataset
prediction = rf.predict_proba(xvalid_bow)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
bow_rf_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR BAG OF WORDS USING RANDOM FOREST\n")
print(bow_rf_pred_score)

#Testing the built RANDOM FOREST model on test data

#predicting on test data
bow_rf_test_pred = rf.predict_proba(test_bow)
    #If probability for label 1 is over 0.3, it is taken as label 1
bow_rf_test_pred_int = bow_rf_test_pred[:,1]>=0.3
bow_rf_test_pred_int = bow_rf_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = bow_rf_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('rf_bow.csv', index=False)





#TF-IDF FEATURES - RANDOM FOREST model
# Training the model with training set from Training dataset
rf.fit(xtrain_tfidf, ytrain)
# prediction on the validation set from Training dataset
prediction = rf.predict_proba(xvalid_tfidf)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
tfidf_rf_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR TF-IDF USING RANDOM FOREST\n")
print(tfidf_rf_pred_score)

#Testing the built RANDOM FOREST on test data

#predicting on test data
tfidf_rf_test_pred = rf.predict_proba(test_tfidf)
    #If probability for label 1 is over 0.3, it is taken as label 1
tfidf_rf_test_pred_int = tfidf_rf_test_pred[:,1]>=0.3
tfidf_rf_test_pred_int = tfidf_rf_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = tfidf_rf_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('rf_tfidf.csv', index=False)





#WORD2VEC FEATURES - RANDOM FOREST model
# Training the model with training set from Training dataset
rf.fit(xtrain_w2v, ytrain)
# prediction on the validation set from Training dataset
prediction = rf.predict_proba(xvalid_w2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
w2v_rf_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR WORD2VEC USING RANDOM FOREST\n")
print(w2v_rf_pred_score)

#Testing the built RANDOM FOREST model on test data

#predicting on test data
w2v_rf_test_pred = rf.predict_proba(test_w2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
w2v_rf_test_pred_int = w2v_rf_test_pred[:,1]>=0.3
w2v_rf_test_pred_int = w2v_rf_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = w2v_rf_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('rf_w2v.csv', index=False)





#DOC2VEC FEATURES - RANDOM FOREST model
# Training the model with training set from Training dataset
rf.fit(xtrain_d2v, ytrain)
# prediction on the validation set from Training dataset
prediction = rf.predict_proba(xvalid_d2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = prediction[:,1]>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
d2v_rf_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR DOC2VEC USING RANDOM FOREST\n")
print(d2v_rf_pred_score)

#Testing the built RANDOM FOREST model on test data

#predicting on test data
d2v_rf_test_pred = rf.predict_proba(test_d2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
d2v_rf_test_pred_int = d2v_rf_test_pred[:,1]>=0.3
d2v_rf_test_pred_int = d2v_rf_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = d2v_rf_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('rf_d2v.csv', index=False)


###################################################################################################################################################################

print("\n\nEXTREME GRADIENT BOOSTING MODEL ACCURACIES\n\n")

#BOW FEATURES - EXTREME GRADIENT DESCENT model
 #Training the model with training set from Training dataset
xgb_cl.fit(xtrain_bow, ytrain)
 #prediction on the validation set from Training dataset
prediction = xgb_cl.predict_proba(xvalid_bow)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = (prediction[:,1])>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
bow_xgb_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR BAG OF WORDS USING EXTREME GRADIENT DESCENT\n")
print(bow_xgb_pred_score)

#Testing the built EXTREME GRADIENT DESCENT model on test data

#predicting on test data
bow_xgb_test_pred = xgb_cl.predict_proba(test_bow)
    #If probability for label 1 is over 0.3, it is taken as label 1
bow_xgb_test_pred_int = (bow_xgb_test_pred[:,1])>=0.3
bow_xgb_test_pred_int = bow_xgb_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = bow_xgb_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('xgb_bow.csv', index=False)





#TF-IDF FEATURES - EXTREME GRADIENT DESCENT model
 #Training the model with training set from Training dataset
xgb_cl.fit(xtrain_tfidf, ytrain)
 #prediction on the validation set from Training dataset
prediction = xgb_cl.predict_proba(xvalid_tfidf)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = (prediction[:,1])>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
tfidf_xgb_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR TF-IDF USING EXTREME GRADIENT DESCENT\n")
print(tfidf_xgb_pred_score)

#Testing the built EXTREME GRADIENT DESCENT on test data

#predicting on test data
tfidf_xgb_test_pred = xgb_cl.predict_proba(test_tfidf)
    #If probability for label 1 is over 0.3, it is taken as label 1
tfidf_xgb_test_pred_int = (tfidf_xgb_test_pred[:,1])>=0.3
tfidf_xgb_test_pred_int = tfidf_xgb_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = tfidf_xgb_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('xgb_tfidf.csv', index=False)





#WORD2VEC FEATURES - EXTREME GRADIENT DESCENT model
# Training the model with training set from Training dataset
xgb_cl.fit(xtrain_w2v, ytrain)
# prediction on the validation set from Training dataset
prediction = xgb_cl.predict(xvalid_w2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = (prediction)>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
w2v_xgb_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR WORD2VEC USING EXTREME GRADIENT DESCENT\n")
print(w2v_xgb_pred_score)

#Testing the built EXTREME GRADIENT DESCENT model on test data

#predicting on test data
w2v_xgb_test_pred = xgb_cl.predict(test_w2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
w2v_xgb_test_pred_int = (w2v_xgb_test_pred)>=0.3
w2v_xgb_test_pred_int = w2v_xgb_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = w2v_xgb_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('xgb_w2v.csv', index=False)





#DOC2VEC FEATURES - EXTREME GRADIENT DESCENT model
# Training the model with training set from Training dataset
xgb_cl.fit(xtrain_d2v, ytrain)
# prediction on the validation set from Training dataset
prediction = xgb_cl.predict_proba(xvalid_d2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
prediction_int = (prediction[:,1])>=0.3
prediction_int = prediction_int.astype(np.int)
#calculating f1 score for model performance
d2v_xgb_pred_score = f1_score(yvalid, prediction_int)
print("\nF1 SCORE FOR DOC2VEC USING EXTREME GRADIENT DESCENT\n")
print(d2v_xgb_pred_score)

#Testing the built EXTREME GRADIENT DESCENT model on test data

#predicting on test data
d2v_xgb_test_pred = xgb_cl.predict_proba(test_d2v)
    #If probability for label 1 is over 0.3, it is taken as label 1
d2v_xgb_test_pred_int = (d2v_xgb_test_pred[:,1])>=0.3
d2v_xgb_test_pred_int = d2v_xgb_test_pred_int.astype(np.int)
#Assigning the predicted values in the label field of test dataset
test['label'] = d2v_xgb_test_pred_int
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('xgb_d2v.csv', index=False)

###################################################################################################################################################################

#FineTuning XGBoost + Word2Vec (as this combo gave better results than other combos)


#Creating Dmatrices for training, validation and testing -> data structure in xgboost; better memory efficiency and training speed
dtrain = xgb.DMatrix(xtrain_w2v, label=ytrain) 
dvalid = xgb.DMatrix(xvalid_w2v, label=yvalid) 
dtest = xgb.DMatrix(test_w2v)


#Defining parameters with default values that we are going to finetune
params = {
    #specifying learing task and laerning objective -> logistic regression for binary classification, output probability
    'objective':'binary:logistic',
    #tree depth ; more the depth, more is the complexity
    'max_depth':6,
    #Minimum no. of samples a node should represent to continue branching; lesser the weight, more is the complexity
    'min_child_weight': 5,
    #Learning rate
    'eta':0.1,
    #XGBoost would randomly sample training data prior to growing trees; sampling-> to select subset from a larger dataset
    'subsample':1,
 }


#creating custom evaluation metric to calculate F1score
def custom_eval(preds, dtrain):
    labels = dtrain.get_label().astype(np.int)
    preds = (preds >= 0.3).astype(np.int)
    return [('f1_score', f1_score(labels, preds))]


#Tuning MAX_DEPTH and MIN_CHILD_WEIGHT
    #Setting different possible combinations of MAX_DEPTH and MIN_CHILD_WEIGHT
gridsearch_params =[(max_depth, min_child_weight)
    for max_depth in range(4,7)
     for min_child_weight in range(4,7)]


# initializing best F1score and best parameters with 0 and None correspondingly
max_f1 = 0.
best_params = None


# Actual fine tuning process for MAX_DEPTH and MIN_CHILD_WEIGHT
for max_depth, min_child_weight in gridsearch_params:
    print("Tuning xgb model with MAX_DEPTH ={}, MIN_CHILD_WEIGHT={}".format(max_depth, min_child_weight))

    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight

    # Cross-validation
    cv_results = xgb.cv(
        params,
        dtrain,
        feval= custom_eval,
        #No of iterations
        num_boost_round=200,
        #To maximize the F1 score
        maximize=True,
        seed=11,
        #No of folds -> divides train data into 5 halves
        nfold=5,
        #error value should decrease atleast 10 iterations once to proceed further. else cross validation ends
        early_stopping_rounds=20
        )

    #Finding max f1 score for this combination of paramaters
    mean_f1 = cv_results['test-f1_score-mean'].max()
    #Finding the round in which max f1 score is reached(idxmax() -> finds index)
    boost_rounds = cv_results['test-f1_score-mean'].idxmax()
    #Displaying the max f1 score and corresponding round
    print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))

    #Computing fmax f1 score for entire combinations conputed so far and reassigning the max f1 and best parameters
    if mean_f1 > max_f1:
        max_f1 = mean_f1
        best_params = (max_depth,min_child_weight)

#Displaying the final best parameter combination of MAX_DEPTH and MIN_CHILD_WEIGHT and corresponding max f1 score
print("Best params: {}, {}, F1 Score: {}".format(best_params[0], best_params[1], max_f1))

#Updating MAX_DEPTH and MIN_CHILD_WEIGHT parameters.

params['max_depth'] = 6
params['min_child_weight'] = 5   

#___________________________________________________________________________________________________________________________________________________#


#Tuning SUBSAMPLE 
    #Setting different possible combinations of SUBSAMPLE (.8 to 1.0)
gridsearch_params = [
    (subsample)
    for subsample in [i/10. for i in range(7,11)] ]

# initializing best F1score and best parameters with 0 and None correspondingly
max_f1 = 0.
best_params = None

# Actual fine tuning process for SUBSAMPLE
for subsample in gridsearch_params:
    print("CV with subsample={}".format(subsample))

    # Update our parameters
    params['subsample'] = subsample

    #Cross-validation
    cv_results = xgb.cv(
        params,
        dtrain,
        feval= custom_eval,
        num_boost_round=200,
        maximize=True,
        seed=11,
        nfold=5,
        early_stopping_rounds=20
    )

    #Finding max f1 score for this combination of paramaters
    mean_f1 = cv_results['test-f1_score-mean'].max()
    #Finding the round in which max f1 score is reached(idxmax() -> finds index)
    boost_rounds = cv_results['test-f1_score-mean'].idxmax()
    #Displaying the max f1 score and corresponding round
    print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))

    #Computing fmax f1 score for entire combinations conputed so far and reassigning the max f1 and best parameters
    if mean_f1 > max_f1:
        max_f1 = mean_f1
        best_params = (subsample)

#Displaying the final best parameter combination of SUBSAMPLE and corresponding max f1 score
print("Best params: {}, F1 Score: {}".format(best_params, max_f1))

#Updating SUBSAMPLE parameters.

params['subsample'] = 0.7

#______________________________________________________________________________________________________________________________________________________#

#Tuning LEARNING RATE 

#Initializing best F1score and best parameters with 0 and None correspondingly
max_f1 = 0.
best_params = None

#Actual fine tuning process for LEARNING RATE
for eta in [.3, .2, .1, .05, .01, .005]:
    print("CV with eta={}".format(eta))

    # Update our parameters
    params['eta'] = eta

    #Cross-validation
    cv_results = xgb.cv(
        params,
        dtrain,
        feval= custom_eval,
        num_boost_round=200,
        maximize=True,
        seed=11,
        nfold=5,
        early_stopping_rounds=20
    )

    #Finding max f1 score for this combination of paramaters
    mean_f1 = cv_results['test-f1_score-mean'].max()
    #Finding the round in which max f1 score is reached(idxmax() -> finds index)
    boost_rounds = cv_results['test-f1_score-mean'].idxmax()
    #Displaying the max f1 score and corresponding round
    print("\tF1 Score {} for {} rounds".format(mean_f1, boost_rounds))

    #Computing fmax f1 score for entire combinations conputed so far and reassigning the max f1 and best parameters
    if mean_f1 > max_f1:
        max_f1 = mean_f1
        best_params = (eta)

#Displaying the final best parameter combination of LEARNING RATE and corresponding max f1 score
print("Best params: {}, F1 Score: {}".format(best_params, max_f1))

#Updating SUBSAMPLE parameters.

params['eta'] = 0.1
   
#______________________________________________________________________________________________________________________________________________________#

print("\n\n\n\n\nF1 SCORES FOR WORD2VEC USING EXTREME GRADIENT DESCENT AFTER FINE TUNING ALL PARAMETERS\n\n\n\n\n")

# XGB Model
xgb_tuned_cl = xgb.train(
    params,
    dtrain,
    feval= custom_eval,
    num_boost_round= 1000,
    maximize=True,
    evals=[(dvalid, "Validation")],
    early_stopping_rounds=20
    )



#Testing the built EXTREME GRADIENT DESCENT model on test data

#predicting on test data
    #Predict_proba is not available for gridsearch params(fine tuned)
w2v_tuned_xgb_test_pred = xgb_tuned_cl.predict(dtest)
    #If probability for label 1 is over 0.3, it is taken as label 1
test['label'] = ( (w2v_tuned_xgb_test_pred) >=0.3 ).astype(np.int)
#Updating id and corresponding predicted labels in new csv file
submission = test[['id','label']]
submission.to_csv('tuned_xgb_w2v.csv', index=False)





#Summary of F1 SCORES for all 4 features using 4 different algorithms
print("\n\nSummary of F1 SCORES for all 4 features using 4 different algorithms\n\n")

lreg_sum = ['   LOGREG ' , "bow: "+str(round(bow_lreg_pred_score,3)) , "tfidf: "+str(round(tfidf_lreg_pred_score,3)) ,
            "w2v: "+str(round(w2v_lreg_pred_score,3)) , "d2v: "+str(round(d2v_lreg_pred_score,3))]

svm_sum =  ['       SVM    ' , "\tbow: "+str(round(bow_svm_pred_score,3)) , "  tfidf: "+str(round(tfidf_svm_pred_score,3)) ,
           "\tw2v: "+str(round(w2v_svm_pred_score,3)) , "\td2v: "+str(round(d2v_svm_pred_score,3))]

rf_sum =   ['           RF     ', "\tbow: "+str(round(bow_rf_pred_score,3)) , "   tfidf: "+str(round(tfidf_rf_pred_score,3)) ,
          "\tw2v: "+str(round(w2v_rf_pred_score,3)) , "\td2v: "+str(round(d2v_rf_pred_score,3))]

xgb_sum =  ['       XGB    ' , "\tbow: "+str(round(bow_xgb_pred_score,3)) , "   tfidf: "+str(round(tfidf_xgb_pred_score,3)) ,
           "\tw2v: "+str(round(w2v_xgb_pred_score,3)) , "\td2v: "+str(round(d2v_lreg_pred_score,3))]

for i,j,k,l in zip(lreg_sum , svm_sum , rf_sum , xgb_sum):
    print(i,j,k,l)

print("\nWORD2VEC F1 SCORES FOR XGB ALGORITHM BEFORE FINE TUNINING IS: {}" .format(w2v_xgb_pred_score))
print("""\n COMAPRE THE F1 SCORE OF BEST ITERATION RETURNED BY FINE TUNED XGB MODEL FOR WORD2VEC AND F1 SCORE OF NORMAL XGB MODEL FOR WORD2VEC (WITHOUT FINE TUNING) TO
NOTICE THE IMPROVEMENT IN ACCURACY""")

