
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from stop_words import get_stop_words
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

stop_words_list = get_stop_words('english')

stop_words_list = stop_words_list + ['aren', 'can', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn',
                                     'haven', 'isn', 'let', 'll', 'mustn', 're', 'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn']

data = pickle.load(open("sklearn-data.pickle", "rb"))

x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test = data["x_test"], data["y_test"]


hv = HashingVectorizer(stop_words=stop_words_list,
                       binary=True, n_features=2**10)

x_train = hv.transform(x_train)
x_test = hv.transform(x_test)

classifierBNB = BernoulliNB()
classifierDTC = DecisionTreeClassifier(max_depth=12)

classifierBNB.fit(X=x_train, y=y_train)
classifierDTC.fit(X=x_train, y=y_train)

train_predicator = classifierBNB.predict(x_train.toarray())
test_predicator = classifierBNB.predict(x_test.toarray())
train_predicator2 = classifierDTC.predict(x_train.toarray())
test_predicator2 = classifierDTC.predict(x_test.toarray())

print("Classifier BernoullI: ")
print("accuracy_score(x_train_pred, y_train): ",
      accuracy_score(train_predicator, y_train))
print("accuracy_score(x_test_pred, y_test): ",
      accuracy_score(test_predicator, y_test))
print('\n')
print("Classifier DecisionTreeCLassifier with max-depth = 12: ")
print("accuracy_score(x_train_pred, y_train): ",
      accuracy_score(train_predicator2, y_train))
print("accuracy_score(x_test_pred, y_test): ",
      accuracy_score(test_predicator2, y_test))
