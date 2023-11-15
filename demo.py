

def demo():
    
    path = 'Restaurant_Reviews.tsv'
    language = 'english' 
    words_to_be_included = ['not','nor','no'] 
    words_to_be_excluded = ['opinion'] 
    no_reviews = 1000
    maximum_features = 1500
    test_set_size = 0.2 
    
    
    import pandas as pd 
    dataset = pd.read_csv(path, delimiter = '\t', quoting = 3) 
    
    
    corpus = review_classification.clean_reviews(no_reviews,language,words_to_be_excluded,words_to_be_included,dataset)   
    # print(type(corpus)) 
    # print(corpus)
    
    X,y = review_classification.bag_of_words(dataset,corpus,maximum_features)
    # print("y",(y[5]))
    # print(X)
    

   
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_set_size)
    
    #Naive Bayes
    accuracy_bayes,score_bayes=review_classification.naive_bayes(X_train,y_train,X_test,y_test)

    #Random Forest
    number_of_trees=1000
    # accuracy_random_forest,score_random_forest=review_classification.random_forest(number_of_trees, X_train,y_train,X_test,y_test)

    #Logistic Regression
    # accuracy_LR,score_LR=review_classification.logistic_regression(X_train,y_train,X_test,y_test)
    
if __name__ == "__main__":
    import review_classification
    demo()