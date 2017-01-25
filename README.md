# Music_classification
NLP/ML Project to songs based on lyrics:

Automatic classification of music based on lyrics in the field of NLP. 
This study investigates the implications of classifying a song based on machine learning techniques specifically, 
Naïve Bayes and SVM classifiers and intends on evaluating their performances, part of this involves the collection of suitably 
large dataset with a good blend of all the genres considered in this study. 
The main goal is to use the results in a recommendation system which can suggest similar songs in the genre to the user. 
This paper finds that there are significant challenges in cleaning the dataset lyrics and also addresses the issues faced 
due to imbalanced dataset and methods used for data balancing

Most of the dataset is borrowed from the MusicMood project[1]. The dataset contains 1000 songs from different genre 
like Rock, Hip Hop/Rap, Jazz ,Country, R&B, Blues, Electronic, Christian and Pop.


[1]https://github.com/rasbt/musicmood

The main.py file takes input.csv as the training set and the testFile.csv is used the test set of songs which needs to be classified.
The results are written in output.txt

Steps followed:

[1]Create Bag of words of input.csv using pandas

[2]Create Bag of words of testFile.csv using pandas

[3]Remove any numbers or space in input or testFile

[4]Create vectors wit CountVectorizer  (simultaneously input is stemmed using PorterStemmer)

[5]fir and transform training file

[6]transform test file

[7]resample training data using RandomOverSampler (imbalanced-learn python library)

[8]Try to fit NaiveBayes

[9]Try to fit SVC
