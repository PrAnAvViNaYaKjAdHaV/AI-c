#Implement decision tree learning algorithm for the restaurant waiting problem.
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
dataset = pd.read_csv('restaurants.csv')
train_features = dataset.iloc[:80,:-1]
test_features = dataset.iloc[80:,:-1]
train_targets = dataset.iloc[:80,-1]
test_targets = dataset.iloc[80:,-1]
tree = DecisionTreeClassifier(criterion = 'entropy').fit(train_features,train_targets)
prediction = tree.predict(test_features)
print("The prediction accuracy is: ",tree.score(test_features,test_targets)*100,"%")
