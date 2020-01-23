# lets import the packeges needed for this

import nltk
import re

# lets import our datasets and specify the columns by using the sep function when importing
import pandas as pd
data = pd.read_csv('formspring_data.csv', sep='\t')

# lets check the number of NANS
data.isnull().any()

# lets now fill our NAN values with false
data.fillna(False)

# lets fish out all the stopwords from the dataset that will not be needed that is basic words like is, and, they etc that cant 
# be used as a feature for cyberbulling

from nltk .corpus import stopwords
stoppers = stopwords.words("English")

print(stoppers)

print("len(stoppers).",len(stoppers))


# now lets now fish out the words that can be used as bully
from nltk.stem.porter import PorterStemmer

ps =PorterStemmer()
corpus = []

for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['post'][i])
    review = review.lower()
    review = review.split()
    
    review  = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# now lets get our independent features which are the words that can be a bully and convert them to array for prediction

from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=5000) #lets use just 5000 from our datasets
x = cv.fit_transform(corpus).toarray()


#lets assign our dependent variable
y = pd.get_dummies(data['bully1'])


# lets import some alglorithms 
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#from sklearn import cross_validation 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets, ensemble

# now lets split our data into test and training set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.33, random_state=0)


# so using random forest alglorithm lets get our result
from sklearn.ensemble import RandomForestClassifier

clf_name=RandomForestClassifier(n_jobs=2, random_state=0)

clf=RandomForestClassifier(n_jobs=2, random_state=0)
#for i,(clf_name,clf) in enumerate(classifiers.items()):
        #fixinng d data and tag outliers

clf.fit(x,y)
y_pred = clf.predict(x)
            #clf.fit(x)
            #scores_pred= clf.decision_function(x)
y_pred = clf.predict(x)
            #reshaping
y_pred[y_pred==1] = 0
y_pred[y_pred== -1] =1
            #cal d num of errors
n_errors = (y_pred != y).sum()
# running classification matrix            
print('{}: {}'.format(clf_name, n_errors))
print("accuracy_score")
print(accuracy_score(y, y_pred))
print("classification_report")
print(classification_report(y, y_pred))