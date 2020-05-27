# K-NEAREST NEIGHBORS(KNN) ON IRIS DATASET

#IMPORTING THE LIBRARIES
from sklearn import datasets
import numpy as np

#IMPORTING IRIS DATASET
iris=datasets.load_iris()
print(iris.data)
print(iris.target)

# SPLITTING DATA INTO TEST AND TRAIN
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(iris.data,iris.target,test_size=0.20)

#FITTING KNN TO THE TRAINING SET
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,y_train)

#IMPORTING ACCURACY MATRIX
from sklearn.metrics import accuracy_score
print("Accuracy is ")
print(accuracy_score(y_test,clf.predict(x_test)))

#PLOTTING A GRAPH WITH K AND ACCURACY
import matplotlib.pyplot as plt
#NOW WE ITERATE OUR CLASSIFIER AND INIT IT DIFFERENT K VALUES AND FIND ACCURACY
#ACCURACY VALUES IS 2D ARRAY[], WHERE EACH ENTRY IS[K,ACCURACY]
accuracy_values=[]
for x in range(1,x_train.shape[0]):
	clf=KNeighborsClassifier(n_neighbors=x).fit(x_train,y_train)
	accuracy=accuracy_score(y_test,clf.predict(x_test))
	accuracy_values.append([x,accuracy])
	pass

#CONVERTING NORMAL PYTHON ARRAY TO NUMPY PYTHON ARRAY FOR SPECIAL OPERATIONS
accuracy_values=np.array(accuracy_values)
plt.plot(accuracy_values[:,0],accuracy_values[:,1])
plt.title('GRAPH WITH K AND ACCURACY')
plt.xlabel("K")
plt.ylabel("ACCURACYy")
plt.show()