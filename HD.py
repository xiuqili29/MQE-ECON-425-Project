import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from matplotlib import pyplot as plt
#BENCHMARK

#Step1. Import data
data = pd.read_csv("425data.csv",names=["Age",'Trestbps','Thalach','Target'])
print(data.head())

# Make default histogram of sepal length
sns.distplot( data["Age"] )

sns.distplot( data["Trestbps"] )

sns.distplot( data["Thalach"] )

 #Step2. Split data into training and test sets
y=data.Target
X=data.drop("Target",axis =1)
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 0.5, random_state = 0)
y_Train = y_Train.ravel()
y_Test = y_Test.ravel()


#Step3. Model Trainning and Evalution
#1. Use sklearn class
clf = LogisticRegression(solver='liblinear')
# Accuracy for Sklearn method:
clf.fit(X_Train,y_Train)
print("Score for Scikit learn: ", clf.score(X_Test,y_Test))


#2. Use KNN method
Acc=[]
for K in range(25):
    Kvalue = K+1
    Knn = KNeighborsClassifier(n_neighbors = Kvalue)
    Knn.fit(X_Train, y_Train)
    y_Pred = Knn.predict(X_Test)
    Acc.append(accuracy_score(y_Test,y_Pred))
    print ("Accuracy is ", accuracy_score(y_Test,y_Pred)," for K-Value:",Kvalue)
Max = max(Acc)
Accuracy=np.array(Acc)
Position=Accuracy.argmax()+1
print ("When K_value is ",Position,", Accuracy is highest, which is ", Max,".")
x = range(1,26)
y = Acc
x=np.array(x)
y=np.array(y)
fig, ax = plt.subplots()
ax.plot(x,y)

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
        arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
annot_max(x,y)
plt.ylabel('Accuracy Rate')
plt.savefig("figure.png")
plt.show()


#Step4. Model Comparsion
Sklearn_score = clf.score(X_Test,y_Test)
Knn_score = Max
if Knn_score > Sklearn_score:
    print ('KNN Method won!')
elif Knn_score == Sklearn_score:
    print ('Its a tie!')
else:
    print ('Scikit won!')
print ('KNN score: ', Knn_score)
print ('Scikits score: ', Sklearn_score)

# Drop Age
#Step1. Import data
data = pd.read_csv("425data.csv",names=["Age",'Trestbps','Thalach','Target'])
data=data.drop("Age",axis =1)
print(data.head())

#Step2. Split data into training and test sets
y=data.Target
X=data.drop("Target",axis =1)
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 0.5, random_state = 0)
y_Train = y_Train.ravel()
y_Test = y_Test.ravel()

#Step3. Model Trainning and Evalution

#1. Use sklearn class
clf = LogisticRegression(solver='liblinear')
# Accuracy for Sklearn method:
clf.fit(X_Train,y_Train)
print("Score for Scikit learn: ", clf.score(X_Test,y_Test))

#2. Use KNN method
Acc=[]
for K in range(25):
    Kvalue = K+1
    Knn = KNeighborsClassifier(n_neighbors = Kvalue)
    Knn.fit(X_Train, y_Train)
    y_Pred = Knn.predict(X_Test)
    Acc.append(accuracy_score(y_Test,y_Pred))
    print ("Accuracy is ", accuracy_score(y_Test,y_Pred)," for K-Value:",Kvalue)
Max = max(Acc)
Accuracy=np.array(Acc)
Position=Accuracy.argmax()+1
print ("When K_value is ",Position,", Accuracy is highest, which is ", Max,".")
x = range(1,26)
y = Acc
x=np.array(x)
y=np.array(y)
fig, ax = plt.subplots()
ax.plot(x,y)
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
        arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
annot_max(x,y)
plt.ylabel('Accuracy Rate')
plt.show()
#Step4. Model Comparsion
Sklearn_score = clf.score(X_Test,y_Test)
Knn_score = Max
if Knn_score > Sklearn_score:
    print ('KNN Method won!')
elif Knn_score == Sklearn_score:
    print ('Its a tie!')
else:
    print ('Scikit won!')
print ('KNN score: ', Knn_score)
print ('Scikits score: ', Sklearn_score)

 # Drop Trestbps
#Step1. Import data
data = pd.read_csv("425data.csv",names=["Age",'Trestbps','Thalach','Target'])
data=data.drop("Trestbps",axis =1)
print(data.head())

#Step2. Split data into training and test sets
y=data.Target
X=data.drop("Target",axis =1)
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 0.5, random_state = 0)
y_Train = y_Train.ravel()
y_Test = y_Test.ravel()

#Step3. Model Trainning and Evalution

#1. Use sklearn class
clf = LogisticRegression(solver='liblinear')
# Accuracy for Sklearn method:
clf.fit(X_Train,y_Train)
print("Score for Scikit learn: ", clf.score(X_Test,y_Test))

#2. Use KNN method
Acc=[]
for K in range(25):
    Kvalue = K + 1
    Knn = KNeighborsClassifier(n_neighbors=Kvalue)
    Knn.fit(X_Train, y_Train)
    y_Pred = Knn.predict(X_Test)
    Acc.append(accuracy_score(y_Test, y_Pred))
    print("Accuracy is ", accuracy_score(y_Test, y_Pred), " for K-Value:", Kvalue)
Max = max(Acc)
Accuracy = np.array(Acc)
Position = Accuracy.argmax() + 1
print("When K_value is ", Position, ", Accuracy is highest, which is ", Max, ".")
x = range(1, 26)
y = Acc
x = np.array(x)
y = np.array(y)
fig, ax = plt.subplots()
ax.plot(x, y)

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
        arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
annot_max(x,y)
plt.ylabel('Accuracy Rate')
plt.show()
#Step4. Model Comparsion
Sklearn_score = clf.score(X_Test,y_Test)
Knn_score = Max
if Knn_score > Sklearn_score:
    print ('KNN Method won!')
elif Knn_score == Sklearn_score:
    print ('Its a tie!')
else:
    print ('Scikit won!')
print ('KNN score: ', Knn_score)
print ('Scikits score: ', Sklearn_score)

# Drop Thalach
#Step1. Import data
data = pd.read_csv("425data.csv",names=["Age",'Trestbps','Thalach','Target'])
data=data.drop("Thalach",axis =1)
print(data.head())
#Step2. Split data into training and test sets
y=data.Target
X=data.drop("Target",axis =1)
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 0.5, random_state = 0)
y_Train = y_Train.ravel()
y_Test = y_Test.ravel()
#Step3. Model Trainning and Evalution
#1. Use sklearn class
clf = LogisticRegression(solver='liblinear')
# Accuracy for Sklearn method:
clf.fit(X_Train,y_Train)
print("Score for Scikit learn: ", clf.score(X_Test,y_Test))
#2. Use KNN method
Acc=[]
for K in range(25):
    Kvalue = K+1
    Knn = KNeighborsClassifier(n_neighbors = Kvalue)
    Knn.fit(X_Train, y_Train)
    y_Pred = Knn.predict(X_Test)
    Acc.append(accuracy_score(y_Test,y_Pred))
    print ("Accuracy is ", accuracy_score(y_Test,y_Pred)," for K-Value:",Kvalue)
Max = max(Acc)
Accuracy=np.array(Acc)
Position=Accuracy.argmax()+1
print ("When K_value is ",Position,", Accuracy is highest, which is ", Max,".")
x = range(1,26)
y = Acc
x=np.array(x)
y=np.array(y)
fig, ax = plt.subplots()
ax.plot(x,y)
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
        arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
annot_max(x,y)
plt.ylabel('Accuracy Rate')
plt.show()
#Step4. Model Comparsion
Sklearn_score = clf.score(X_Test,y_Test)
Knn_score = Max
if Knn_score > Sklearn_score:
    print ('KNN Method won!')
elif Knn_score == Sklearn_score:
    print ('Its a tie!')
else:
    print ('Scikit won!')
print ('KNN score: ', Knn_score)
print ('Scikits score: ', Sklearn_score)
# Drop Age and Trestbps
#Step1. Import data
data = pd.read_csv("425data.csv",names=["Age",'Trestbps','Thalach','Target'])
data=data.drop(["Age","Trestbps"],axis =1)
print(data.head())
#Step2. Split data into training and test sets
y=data.Target
X=data.drop("Target",axis =1)
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 0.5, random_state = 0)
y_Train = y_Train.ravel()
y_Test = y_Test.ravel()
#Step3. Model Trainning and Evalution
#1. Use sklearn class
clf = LogisticRegression(solver='liblinear')
# Accuracy for Sklearn method:
clf.fit(X_Train,y_Train)
print("Score for Scikit learn: ", clf.score(X_Test,y_Test))
#2. Use KNN method
Acc=[]
for K in range(25):
    Kvalue = K+1
    Knn = KNeighborsClassifier(n_neighbors = Kvalue)
    Knn.fit(X_Train, y_Train)
    y_Pred = Knn.predict(X_Test)
    Acc.append(accuracy_score(y_Test,y_Pred))
    print ("Accuracy is ", accuracy_score(y_Test,y_Pred)," for K-Value:",Kvalue)
Max = max(Acc)
Accuracy=np.array(Acc)
Position=Accuracy.argmax()+1
print ("When K_value is ",Position,", Accuracy is highest, which is ", Max,".")
x = range(1,26)
y = Acc
x=np.array(x)
y=np.array(y)
fig, ax = plt.subplots()
ax.plot(x,y)
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
        arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
annot_max(x,y)
plt.ylabel('Accuracy Rate')
plt.show()
#Step4. Model Comparsion
Sklearn_score = clf.score(X_Test,y_Test)
Knn_score = Max
if Knn_score > Sklearn_score:
    print ('KNN Method won!')
elif Knn_score == Sklearn_score:
    print ('Its a tie!')
else:
    print ('Scikit won!')
print ('KNN score: ', Knn_score)
print ('Scikits score: ', Sklearn_score)
# Drop Age and Thalach
#Step1. Import data
data = pd.read_csv("425data.csv",names=["Age",'Trestbps','Thalach','Target'])
data=data.drop(["Age","Thalach"],axis =1)
print(data.head())
#Step2. Split data into training and test sets
y=data.Target
X=data.drop("Target",axis =1)
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 0.5, random_state = 0)
y_Train = y_Train.ravel()
y_Test = y_Test.ravel()
#Step3. Model Trainning and Evalution
#1. Use sklearn class
clf = LogisticRegression(solver='liblinear')
# Accuracy for Sklearn method:
clf.fit(X_Train,y_Train)
print("Score for Scikit learn: ", clf.score(X_Test,y_Test))
#2. Use KNN method
Acc=[]
for K in range(25):
    Kvalue = K+1
    Knn = KNeighborsClassifier(n_neighbors = Kvalue)
    Knn.fit(X_Train, y_Train)
    y_Pred = Knn.predict(X_Test)
    Acc.append(accuracy_score(y_Test,y_Pred))
    print ("Accuracy is ", accuracy_score(y_Test,y_Pred)," for K-Value:",Kvalue)
Max = max(Acc)
Accuracy=np.array(Acc)
Position=Accuracy.argmax()+1
print ("When K_value is ",Position,", Accuracy is highest, which is ", Max,".")
x = range(1,26)
y = Acc
x=np.array(x)
y=np.array(y)
fig, ax = plt.subplots()
ax.plot(x,y)
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
        arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
annot_max(x,y)
plt.ylabel('Accuracy Rate')
plt.show()
#Step4. Model Comparsion
Sklearn_score = clf.score(X_Test,y_Test)
Knn_score = Max
if Knn_score > Sklearn_score:
    print ('KNN Method won!')
elif Knn_score == Sklearn_score:
    print ('Its a tie!')
else:
    print ('Scikit won!')
print ('KNN score: ', Knn_score)
print ('Scikits score: ', Sklearn_score)

# Drop Trestbps and Thalach
#Step1. Import data
data = pd.read_csv("425data.csv",names=["Age",'Trestbps','Thalach','Target'])
data=data.drop(['Trestbps',"Thalach"],axis =1)
print(data.head())
#Step2. Split data into training and test sets
y=data.Target
X=data.drop("Target",axis =1)
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 0.5, random_state = 0)
y_Train = y_Train.ravel()
y_Test = y_Test.ravel()
#Step3. Model Trainning and Evalution
#1. Use sklearn class
clf = LogisticRegression(solver='liblinear')
# Accuracy for Sklearn method:
clf.fit(X_Train,y_Train)
print("Score for Scikit learn: ", clf.score(X_Test,y_Test))
#2. Use KNN method
Acc=[]
for K in range(25):
    Kvalue = K+1
    Knn = KNeighborsClassifier(n_neighbors = Kvalue)
    Knn.fit(X_Train, y_Train)
    y_Pred = Knn.predict(X_Test)
    Acc.append(accuracy_score(y_Test,y_Pred))
    print ("Accuracy is ", accuracy_score(y_Test,y_Pred)," for K-Value:",Kvalue)
Max = max(Acc)
Accuracy=np.array(Acc)
Position=Accuracy.argmax()+1
print ("When K_value is ",Position,", Accuracy is highest, which is ", Max,".")
x = range(1,26)
y = Acc
x=np.array(x)
y=np.array(y)
fig, ax = plt.subplots()
ax.plot(x,y)
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
        arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
annot_max(x,y)
plt.ylabel('Accuracy Rate')
plt.show()
#Step4. Model Comparsion
Sklearn_score = clf.score(X_Test,y_Test)
Knn_score = Max
if Knn_score > Sklearn_score:
    print ('KNN Method won!')
elif Knn_score == Sklearn_score:
    print ('Its a tie!')
else:
    print ('Scikit won!')
print ('KNN score: ', Knn_score)
print ('Scikits score: ', Sklearn_score)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
24# test size = 0.1
#Step1. Import data
data = pd.read_csv("425data.csv",names=["Age",'Trestbps','Thalach','Target'])
print(data.head())
#Step2. Split data into training and test sets
y=data.Target
X=data.drop("Target",axis =1)
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 0.1, random_state = 0)
y_Train = y_Train.ravel()
y_Test = y_Test.ravel()
#Step3. Model Trainning and Evalution
#1. Use sklearn class
clf = LogisticRegression(solver='liblinear')
# Accuracy for Sklearn method:
clf.fit(X_Train,y_Train)
print("Score for Scikit learn: ", clf.score(X_Test,y_Test))
#2. Use KNN method
Acc=[]
for K in range(25):
    Kvalue = K + 1
    Knn = KNeighborsClassifier(n_neighbors=Kvalue)
    Knn.fit(X_Train, y_Train)
    y_Pred = Knn.predict(X_Test)
    Acc.append(accuracy_score(y_Test, y_Pred))
    print("Accuracy is ", accuracy_score(y_Test, y_Pred), " for K-Value:", Kvalue)
Max = max(Acc)
Accuracy = np.array(Acc)
Position = Accuracy.argmax() + 1
print("When K_value is ", Position, ", Accuracy is highest, which is ", Max, ".")
x = range(1, 26)
y = Acc
x = np.array(x)
y = np.array(y)

fig, ax = plt.subplots()
ax.plot(x, y)


def annot_max(x, y, ax=None):
    xmax = x[np.argmax(y)]


    ymax = y.max()
    text = "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)
annot_max(x, y)
plt.ylabel('Accuracy Rate')
plt.show()
# Step4. Model Comparsion
Sklearn_score = clf.score(X_Test, y_Test)
Knn_score = Max
if Knn_score > Sklearn_score:
    print('KNN Method won!')
elif Knn_score == Sklearn_score:
    print('Its a tie!')
else:
    print('Scikit won!')
print('KNN score: ', Knn_score)
print('Scikits score: ', Sklearn_score)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#test size = 0.2
#Step1. Import data
data = pd.read_csv("425data.csv",names=["Age",'Trestbps','Thalach','Target'])
print(data.head())
#Step2. Split data into training and test sets
y=data.Target
X=data.drop("Target",axis =1)
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_Train = y_Train.ravel()
y_Test = y_Test.ravel()
#Step3. Model Trainning and Evalution
#1. Use sklearn class
clf = LogisticRegression(solver='liblinear')
# Accuracy for Sklearn method:
clf.fit(X_Train,y_Train)
print("Score for Scikit learn: ", clf.score(X_Test,y_Test))
#2. Use KNN method
Acc=[]
for K in range(25):
    Kvalue = K+1
    Knn = KNeighborsClassifier(n_neighbors = Kvalue)
    Knn.fit(X_Train, y_Train)
    y_Pred = Knn.predict(X_Test)
    Acc.append(accuracy_score(y_Test,y_Pred))
    print ("Accuracy is ", accuracy_score(y_Test,y_Pred)," for K-Value:",Kvalue)
Max = max(Acc)
Accuracy=np.array(Acc)
Position=Accuracy.argmax()+1
print ("When K_value is ",Position,", Accuracy is highest, which is ", Max,".")
x = range(1,26)
y = Acc
x=np.array(x)
y=np.array(y)
fig, ax = plt.subplots()
ax.plot(x,y)
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
        arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
annot_max(x,y)
plt.ylabel('Accuracy Rate')
plt.show()
#Step4. Model Comparsion
Sklearn_score = clf.score(X_Test,y_Test)
Knn_score = Max
if Knn_score > Sklearn_score:
    print ('KNN Method won!')
elif Knn_score == Sklearn_score:
    print ('Its a tie!')
else:
    print ('Scikit won!')
print ('KNN score: ', Knn_score)
print ('Scikits score: ', Sklearn_score)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#Test size = 0.3
#Step1. Import data
data = pd.read_csv("425data.csv",names=["Age",'Trestbps','Thalach','Target'])
print(data.head())
#Step2. Split data into training and test sets
y=data.Target
X=data.drop("Target",axis =1)
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 0.3, random_state = 0)
y_Train = y_Train.ravel()
y_Test = y_Test.ravel()
#Step3. Model Trainning and Evalution
#1. Use sklearn class
clf = LogisticRegression(solver='liblinear')
# Accuracy for Sklearn method:
clf.fit(X_Train,y_Train)
print("Score for Scikit learn: ", clf.score(X_Test,y_Test))
#2. Use KNN method
Acc=[]
for K in range(25):
    Kvalue = K+1
    Knn = KNeighborsClassifier(n_neighbors = Kvalue)
    Knn.fit(X_Train, y_Train)
    y_Pred = Knn.predict(X_Test)
    Acc.append(accuracy_score(y_Test,y_Pred))
    print ("Accuracy is ", accuracy_score(y_Test,y_Pred)," for K-Value:",Kvalue)
Max = max(Acc)
Accuracy=np.array(Acc)
Position=Accuracy.argmax()+1
print ("When K_value is ",Position,", Accuracy is highest, which is ", Max,".")
x = range(1,26)
y = Acc
x=np.array(x)
y=np.array(y)
fig, ax = plt.subplots()
ax.plot(x,y)
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
        arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
annot_max(x,y)
plt.ylabel('Accuracy Rate')
plt.show()
#Step4. Model Comparsion
Sklearn_score = clf.score(X_Test,y_Test)
Knn_score = Max
if Knn_score > Sklearn_score:
    print ('KNN Method won!')
elif Knn_score == Sklearn_score:
    print ('Its a tie!')
else:
    print ('Scikit won!')
print ('KNN score: ', Knn_score)
print ('Scikits score: ', Sklearn_score)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#test size = 0.4
#Step1. Import data
data = pd.read_csv("425data.csv",names=["Age",'Trestbps','Thalach','Target'])
print(data.head())
#Step2. Split data into training and test sets
y=data.Target
X=data.drop("Target",axis =1)
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size = 0.4, random_state = 0)
y_Train = y_Train.ravel()
y_Test = y_Test.ravel()
#Step3. Model Trainning and Evalution
#1. Use sklearn class
clf = LogisticRegression(solver='liblinear')
# Accuracy for Sklearn method:
clf.fit(X_Train,y_Train)
print("Score for Scikit learn: ", clf.score(X_Test,y_Test))
#2. Use KNN method
Acc=[]
for K in range(25):
    Kvalue = K+1
    Knn = KNeighborsClassifier(n_neighbors = Kvalue)
    Knn.fit(X_Train, y_Train)
    y_Pred = Knn.predict(X_Test)
    Acc.append(accuracy_score(y_Test,y_Pred))
    print ("Accuracy is ", accuracy_score(y_Test,y_Pred)," for K-Value:",Kvalue)
Max = max(Acc)
Accuracy=np.array(Acc)
Position=Accuracy.argmax()+1
print ("When K_value is ",Position,", Accuracy is highest, which is ", Max,".")
x = range(1,26)
y = Acc
x=np.array(x)
y=np.array(y)
fig, ax = plt.subplots()
ax.plot(x,y)
def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
        arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
annot_max(x,y)
plt.ylabel('Accuracy Rate')
plt.show()
#Step4. Model Comparsion
Sklearn_score = clf.score(X_Test,y_Test)
Knn_score = Max
if Knn_score > Sklearn_score:
    print ('KNN Method won!')
elif Knn_score == Sklearn_score:
    print ('Its a tie!')
else:
    print ('Scikit won!')
print ('KNN score: ', Knn_score)
print ('Scikits score: ', Sklearn_score)
# fitting SVM model
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
svm = svm.SVC(C = 1000, kernel = "linear")
svm.fit(X_Train, y_Train)
print("Score for SVM:", svm.score(X_Test,y_Test))
svmy_pred = svm.predict(X_Test)
# error of random forest
svmYDiff = np.abs(svmy_pred - y_Test)
svm_avgerror_rf = np.mean(svmYDiff)
svm_stderror_rf = np.std(svmYDiff)
print("Avg Error SVM:", svm_avgerror_rf)
print("Std Error SVM:", svm_stderror_rf)
print(confusion_matrix(y_Test,svmy_pred))
print(classification_report(y_Test,svmy_pred))
print(accuracy_score(y_Test, svmy_pred))

# fitting random forest model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=20, random_state=0)
rf.fit(X_Train, y_Train)
print("Score for RandomForestClassifier:", rf.score(X_Test,y_Test))
rfy_pred = rf.predict(X_Test)
# error of random forest
rfYDiff = np.abs(rfy_pred - y_Test)
rf_avgerror_rf = np.mean(rfYDiff)
rf_stderror_rf = np.std(rfYDiff)
print("Avg Error RF:", rf_avgerror_rf)
print("Std Error RF:", rf_stderror_rf)
print(confusion_matrix(y_Test,rfy_pred))
print(classification_report(y_Test,rfy_pred))
print(accuracy_score(y_Test, rfy_pred))
# comparing SVM and Random Forest Methods
SVM_score = svm.score(X_Test,y_Test)
RF_score = rf.score(X_Test,y_Test)
if SVM_score > RF_score:
    print ('SVM method won!')
elif SVM_score == RF_score:
    print ('Its a tie!')
else:
    print ('Random Forest won!')
print ("SVM Score:", SVM_score)
print ("RF Score:", RF_score)
# Stochastic Gradient Descent with loss functions of logistic regression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
SGD2 = SGDClassifier(loss="log")
SGD2.fit(X_Train, y_Train)
SGD2_pred =SGD2.predict(X_Test)
print(confusion_matrix(y_Test,SGD2_pred))
print(classification_report(y_Test,SGD2_pred))
print('accuracy:',accuracy_score(y_Test,SGD2_pred))
# Nearest Centroid Classifier
from sklearn.neighbors._nearest_centroid import NearestCentroid
NCC = NearestCentroid()
NCC.fit(X_Train, y_Train)
NCC_pred=NCC.predict(X_Test)
print(confusion_matrix(y_Test,NCC_pred))
print(classification_report(y_Test,NCC_pred))
print('accuracy:',accuracy_score(y_Test,NCC_pred))
