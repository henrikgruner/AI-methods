
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from stop_words import get_stop_words
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import accuracy_score
data = pickle.load(open("sklearn-data.pickle", "rb"))

x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test = data["x_test"], data["y_test"]

stop_words = get_stop_words('english')

stop_words = stop_words + ['aren', 'can', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven',
                           'isn', 'let', 'll', 'mustn', 're', 'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn']


hv = HashingVectorizer(
    stop_words=['stop', 'is'], binary=True, n_features=2**10)


hv.fit_transform(x_train)
print(len(x_train))
print(len(y_train))
hv.transform(x_test)


classifier = BernoulliNB()


classifier.fit(X=(x_train, y_train), y=y_test)

train_pred = classifier.predict(x_train.toarray())
test_pred = classifier.predict(x_test.toarray())

print(train_acc=accuracy_score(y_train, train_pred))
print(test_acc=accuracy_score(y_test, test_pred))
