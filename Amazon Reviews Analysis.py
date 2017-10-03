import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD


# read the entire file into a Python array
with open('C:/Users/Nathan.Mannheimer/Desktop/Data Science Club/Amazon Reviews/Musical_Instruments_5.json', 'r') as f:
    # Extract each line
    data = (line.strip() for line in f)
    # Reformat so each line is the element of a list
    data_json = "[{0}]".format(','.join(data))
# read the result as a JSON
df = pd.read_json(data_json)
print(df.head())

y = df.overall
vect = CountVectorizer(stop_words='english', ngram_range=(1,2), analyzer='word')
X = vect.fit_transform(df.reviewText)

print(vect.get_feature_names().index('love'))
print(X[:,146009].sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)

pca = TruncatedSVD(n_components=1500)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
svc_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

svc = SVC(class_weight='balanced')
model = GridSearchCV(svc, svc_parameters,cv=kf, scoring='accuracy', verbose=True)
#model = SVC(class_weight='balanced', C=10, kernel='rbf', gamma= 1e-4)
model.fit(X_train, y_train)


print(classification_report(y_test, model.predict(X_test)))
print(model.score(X_test,y_test))
