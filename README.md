# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students
2. Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored
3. Define and initialize the parameters for regression model: slope  controls the steepness and intercept represents where the line crsses the y-axis
4. Use the linear equation to predict marks based on the input
   Predicted Marks = m.(hours studied) + b
5. for each data point calculate the difference between the actual and predicted marks
6. Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error
7. Once the model parameters are optimized, use the final equation to predict marks for any new input data
   
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Preethi S
RegisterNumber: 212223230157

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('student_scores.csv')
print(df)
print()
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse) 
*/
```


## Output:
![image](https://github.com/user-attachments/assets/f8b94176-f7c8-444b-b67b-b1b679af64f7)
![image](https://github.com/user-attachments/assets/a7a3802b-d4b1-48d1-9526-6bc97d784b87)
![image](https://github.com/user-attachments/assets/2ea20f5f-8455-4037-8999-3edada69a9e2)
![image](https://github.com/user-attachments/assets/a2b3f7c0-4e83-47fe-b9f5-82db5b8131c2)
![image](https://github.com/user-attachments/assets/adc5f0d6-c0ba-4cc5-9ce1-3bc13a5faf13)
![image](https://github.com/user-attachments/assets/76da8792-b4dc-47aa-b6f6-34b4630d1980)
![image](https://github.com/user-attachments/assets/b68c688b-35c6-4181-8c87-1dc8fba8f604)
![image](https://github.com/user-attachments/assets/7385981a-dacc-4989-909a-d1012e9599be)
![image](https://github.com/user-attachments/assets/19831476-0423-4764-83af-0d4da74804af)
![image](https://github.com/user-attachments/assets/21aa32c3-29f6-4e10-973b-d13cad4a928b)
![image](https://github.com/user-attachments/assets/92c19276-8378-44d1-8acf-3b5ee00e01da)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
