import numpy as np
import pandas as pd
import sklearn
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

###################################### Credit: LogisticRegression.ipynb from session ##################################################
#I used the median age of Pclass=3 as a replacement as there is no median value for SibSp=8 in training dataset
def fill_age(dataset,dataset_med):
    for x in range(len(dataset)):
        if dataset["Pclass"][x]==1:
            if dataset["SibSp"][x]==0:
                return dataset_med.loc[1,0]["Age"]
            elif dataset["SibSp"][x]==1:
                return dataset_med.loc[1,1]["Age"]
            elif dataset["SibSp"][x]==2:
                return dataset_med.loc[1,2]["Age"]
            elif dataset["SibSp"][x]==3:
                return dataset_med.loc[1,3]["Age"]
        elif dataset["Pclass"][x]==2:
            if dataset["SibSp"][x]==0:
                return dataset_med.loc[2,0]["Age"]
            elif dataset["SibSp"][x]==1:
                return dataset_med.loc[2,1]["Age"]
            elif dataset["SibSp"][x]==2:
                return dataset_med.loc[2,2]["Age"]
            elif dataset["SibSp"][x]==3:
                return dataset_med.loc[2,3]["Age"]
        elif dataset["Pclass"][x]==3:
            if dataset["SibSp"][x]==0:
                return dataset_med.loc[3,0]["Age"]
            elif dataset["SibSp"][x]==1:
                return dataset_med.loc[3,1]["Age"]
            elif dataset["SibSp"][x]==2:
                return dataset_med.loc[3,2]["Age"]
            elif dataset["SibSp"][x]==3:
                return dataset_med.loc[3,3]["Age"]
            elif dataset["SibSp"][x]==4:
                return dataset_med.loc[3,4]["Age"]
            elif dataset["SibSp"][x]==5:
                return dataset_med.loc[3,5]["Age"]
            elif dataset["SibSp"][x]==8:
                return dataset_med.loc[3]["Age"].median()

#Cabin U is when the rest of cabins are 0
def new_cabin_features(dataset):
    dataset["Cabin A"]=np.where(dataset["Cabin"]=="A",1,0)
    dataset["Cabin B"]=np.where(dataset["Cabin"]=="B",1,0)
    dataset["Cabin C"]=np.where(dataset["Cabin"]=="C",1,0)
    dataset["Cabin D"]=np.where(dataset["Cabin"]=="D",1,0)
    dataset["Cabin E"]=np.where(dataset["Cabin"]=="E",1,0)
    dataset["Cabin F"]=np.where(dataset["Cabin"]=="F",1,0)
    dataset["Cabin G"]=np.where(dataset["Cabin"]=="G",1,0)
    dataset["Cabin T"]=np.where(dataset["Cabin"]=="T",1,0)

########################################################################################################################

titanic = pd.read_csv('titanic.csv')
titanic_1 = titanic.groupby(["Pclass","SibSp"])
titanic_1_median = titanic_1.median()
titanic["Age"] = titanic["Age"].fillna(fill_age(titanic,titanic_1_median))
titanic["Cabin"] = titanic["Cabin"].fillna("U")
titanic["Cabin"] = titanic["Cabin"].map(lambda x: x[0])
new_cabin_features(titanic)

x = titanic.drop(['Name','Cabin','Embarked','Ticket','Survived'], axis=1)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')
x=(np.array(ct.fit_transform(x)))
y=titanic["Survived"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

########################################### PLEASE COMMENT OUT THE UNECESSARY KERNEL BEFORE RUNNING ###################################
### All hyperparameters obtained from GridSearchCV and some guessing (XD)
### Linear Kernel ###
classifier = SVC(kernel = 'linear', C = 10)
classifier.fit(x_train, y_train)
predicted_test = classifier.predict(x_test)
predicted_train = classifier.predict(x_train)


### RBF Kernel ###
classifier = SVC(kernel = 'rbf', C = 1000, gamma=0.0001)
classifier.fit(x_train, y_train)
predicted_test = classifier.predict(x_test)
predicted_train = classifier.predict(x_train)

########################################### PLEASE COMMENT OUT METRICS FOR MODEL IF NOT NEEDED #########################################

### Metrics for model ###
print('Accuracy Score of Test: ' + str(accuracy_score(y_test,predicted_test)))
print('Precision Score of Test: ' + str(precision_score(y_test,predicted_test)))
print('Recall Score of Test: ' + str(recall_score(y_test,predicted_test)))
print('F1 Score of Test: ' + str(f1_score(y_test, predicted_test)))
print('Confusion Matrix of Test: \n' + str(confusion_matrix(y_test,predicted_test)))

print('Accuracy Score of Train: ' + str(accuracy_score(y_train,predicted_train)))
print('Precision Score of Train: ' + str(precision_score(y_train,predicted_train)))
print('Recall Score of Train: ' + str(recall_score(y_train,predicted_train)))
print('F1 Score of Train: ' + str(f1_score(y_train, predicted_train)))
print('Confusion Matrix of Train: \n' + str(confusion_matrix(y_train,predicted_train)))


################################################# PLEASE USE FILE NAME CORRESPONDING TO KERNEL USED ###################################

### Save kernel output as .csv file ###
###labels_df = pd.DataFrame(predicted_test,columns=["Survived"])
###labels_df.to_csv(r'C:\Users\MelPr\PycharmProjects\pythonProject1\LinearKernel.csv',index = False)

