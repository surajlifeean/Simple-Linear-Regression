
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :1].values
y = dataset.iloc[:, 1].values


#splitting data into training set and test set

from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

# fitting sample linear regression data into training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the data based on test set

y_pred=regressor.predict(X_test)
y_p=regressor.predict(1.42)
#visualizing data

plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary VS Experience')
plt.show()