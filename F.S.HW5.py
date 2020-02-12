import pandas as pd
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np

#Import data and add column labels
data = pd.read_csv('/home/fubunutu/PycharmProjects/hw5/adult.data', header=None, index_col=False, names=['age', 'workclass', 'fnlwgt', 'education',
                                                                      'education-num', 'marital-status', 'occupation',
                                                                      'relationship', 'race', 'gender', 'capital-gain',
                                                                      'capital-loss', 'hours-per-week', 'native-country',
                                                                      'income'])




#Print features before and after one hot encoding with Pandas dummies
print('Original Labels:\n', list(data.columns), '\n')
data_dummies = pd.get_dummies(data)
print('Labels after One-Hot Encoding with Pandas dummies:\n', list(data_dummies.columns))




#Selecting all collumns for x values except income
features = data_dummies.ix[:, 'age':'native-country_ Yugoslavia']
X = features.values

#Setting y values = income > 50k, we will be predicting if income is over 50k
y = data_dummies['income_ >50k'].values

print(data_dummies['income_ <= 50k'].values)

#split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.30)

#create random forest classfier var
RF = RandomForestClassifier(n_jobs=-1)

#create SVM classifier var
svmmodel = svm.SVC(gamma='scale', degree=2)
#fit data to models
svmmodel.fit(X_train, y_train)
RF.fit(X_train, y_train)

#print accuracy scores for SVM and Random forest
print('Random forest score on the test set: {:.2f}'.format(RF.score(X_test, y_test)))
print('SVM score on the test set: {:.2f}'.format(svmmodel.score(X_test, y_test)))


y_predSVM = svmmodel.predict(X_test)
y_predRF = RF.predict(X_test)


import itertools

#Create confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#plots confusion matrix normalized
cm = metrics.confusion_matrix(y_test, y_predSVM)
plot_confusion_matrix(cm, ['0', '1'], normalize=True, title='SVM confusion matrix.')
plt.show()

cm2 = metrics.confusion_matrix(y_test, y_predRF)
plot_confusion_matrix(cm2, ['0', '1'], normalize=True, title='Random forest confusion matrix.')
plt.show()