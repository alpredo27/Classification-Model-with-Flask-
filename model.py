
import pandas as pd
import numpy as np
import pickle as Pickle

#get data and save to data frame
df = pd.read_csv('bank.csv', delimiter = ';')

#Split the data - Input and Output variables
y = df.y
X = df.drop(['y'], axis = 1)

#For simplicity, only consider the first 3 numeric variables
numeric = ['age','balance','day','duration','campaign','pdays', 'previous']
numeric = numeric[:3]

#Model Build
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e5)
logreg.fit(X[numeric], y)

#save the model to pickle file
Pickle.dump(logreg, open('model.pkl','wb'))



#sample prediction using the pickle file to test it out
model = Pickle.load(open('model.pkl','rb'))
a = 2
b = 4
c = 5
print(model.predict([[a, b, c]]))