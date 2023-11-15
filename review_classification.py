
#Cleaning the reviews
def clean_reviews(no_reviews,language, words_to_be_excluded, words_to_be_included, dataset):
    import re
    import nltk
    # nltk.download('stopwords') #Must only be downloaded once 
    from nltk.corpus import stopwords 
    
    
    my_stopwords = stopwords.words(language) 
   
    for word in words_to_be_excluded:
        if not any ( word in stpwords for stpwords in my_stopwords):
            my_stopwords.append(word)
    
    
    for word in words_to_be_included:
        if any (word in stpwords for stpwords in my_stopwords):
            my_stopwords.remove(word)
            
    
    from nltk.stem.porter import PorterStemmer
     
    corpus =[] 
    for i in range (0, no_reviews):
        review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i]) 
        review = review.lower() 
        review = review.split() 
        ps = PorterStemmer() 
        review = [ps.stem(word) for word in review if not word in set (my_stopwords)] 
        review = ' '.join(review) 
        corpus.append(review) 
    return corpus

#Creating the bag of words model
def bag_of_words(dataset, corpus, maximum_features):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features = maximum_features) 
    X = cv.fit_transform(corpus).toarray() #Sparse matrix 
    # print(cv.get_feature_names_out())
    # print(X[1])
    print(len(X[1]))
    y = dataset.iloc[:,1].values 
    return X,y

def score_acc(cm):
    TN= cm[0][0] 
    TP = cm[1][1]
    
    FN = cm[0][1] 
    FP = cm [1][0]
    accuracy = (TP + TN)/(TP + TN + FP + FN) 
    precision = TP / (TP + FP) 
    recall = TP / (TP + FN)
    score = 2 * precision * recall/(precision + recall)
    return accuracy,score

#Naive Bayes classification algorithm
def naive_bayes(X_train, y_train, X_test, y_test):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    # print(X_test)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)   
    # print(y_pred)
    
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred) #Contains results of predictions
    print(cm)
    accuracy,score=score_acc(cm)
    print("Accuracy of the Naive Bayes algorithm is ",  accuracy, " and score ", score)
    return accuracy, score    


#Random Forest classification algorithm
def random_forest(number_of_trees, X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = number_of_trees, criterion = 'entropy')
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy,score=score_acc(cm)
    print("Accuracy of the Random Forest algorithm for ", number_of_trees, " is ",  accuracy, " and score ", score)
    return accuracy, score
      
#Logistic Regression classification algorithm
def logistic_regression(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy,score=score_acc(cm)
    print("Accuracy of the Logistic Regression algorithm is ",  accuracy, " and score ", score)
    return accuracy, score


    
