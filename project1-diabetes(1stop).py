#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


dataset=pd.read_csv("diabetes.csv")
dataset.info()


# In[4]:


dataset.isnull().sum()


# In[5]:



dataset.describe()


# In[7]:


#correlation plot of independent variables
plt.figure(figsize=(10,8))
sns.heatmap(dataset.corr(),annot=True,fmt=".3f",cmap="YlGnBu")
plt.title("correlation map")


# In[12]:


#exploring pregnancies and target values
plt.figure(figsize=(10,8))
kde=sns.kdeplot(dataset["Pregnancies"][dataset["Outcome"]==1],color="Red",shade=True)
kde=sns.kdeplot(dataset["Pregnancies"][dataset["Outcome"]==0],color="Green",shade=True)
kde.set_xlabel("Pregnancies")
kde.set_ylabel("Density")
kde.legend(["Positive","Negative"])


# In[13]:


#Eploring glucose and target variables
plt.figure(figsize=(10,8))
sns.violinplot(data=dataset,x="Outcome",y="Glucose",split=True,linewidth=2,inner="quart")


# In[20]:


#densityplot for glucose and target variables
plt.figure(figsize=(10,8))
kde=sns.kdeplot(dataset["Glucose"][dataset["Outcome"]==1],color="Red",shade=True)
kde=sns.kdeplot(dataset["Glucose"][dataset["Outcome"]==0],color="purple",shade=True)
kde.set_xlabel("glucose")
kde.set_ylabel("Density")
kde.legend(["Positive","Negative"])
#Replacing 0 values with mean or median values of the respective feature
dataset["Glucose"]=dataset["Glucose"].replace(0,dataset["Glucose"].median())
dataset["BloodPressure"]=dataset["BloodPressure"].replace(0,dataset["BloodPressure"].median())
dataset["BMI"]=dataset["BMI"].replace(0,dataset["BMI"].median())
dataset["SkinThickness"]=dataset["SkinThickness"].replace(0,dataset["SkinThickness"].median())
dataset["Insulin"]=dataset["Insulin"].replace(0,dataset["Insulin"].median())


# In[24]:


x=dataset.drop(["Outcome"],axis=1)
y=dataset["Outcome"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
x_train


# In[32]:


from sklearn.neighbors import KNeighborsClassifier
training_accuracy=[]
testing_accuracy=[]
for n_neighbors in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train,y_train)
    #check accuracy
    training_accuracy.append(knn.score(x_train,y_train))
    testing_accuracy.append(knn.score(x_test,y_test))
plt.plot(range(1,11),training_accuracy,label="training_accuracy")
plt.plot(range(1,11),testing_accuracy,label="test_accuracy")
plt.xlabel("n_neighbors")
plt.ylabel("accuracy")
plt.show()


# In[34]:


knn=KNeighborsClassifier(n_neighbors)
knn.fit(x_train,y_train)
print(knn.score(x_train,y_train),":Training accuracy")
print(knn.score(x_test,y_test),":Test accuracy")


# In[36]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=0)
dt.fit(x_train,y_train)
print(dt.score(x_train,y_train),":Training accuracy")
print(dt.score(x_test,y_test),":Test accuracy")


# In[37]:


dt1=DecisionTreeClassifier(random_state=0,max_depth=3)
dt1.fit(x_train,y_train)
print(dt1.score(x_train,y_train),":Training accuracy")
print(dt1.score(x_test,y_test),":Test accuracy")


# In[39]:


from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(random_state=0)
mlp.fit(x_train,y_train)
print(mlp.score(x_train,y_train),":Training accuracy")
print(mlp.score(x_test,y_test),":Test accuracy")


# In[41]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train_scaled=sc.fit_transform(x_train)
x_test_scaled=sc.fit_transform(x_test)


# In[42]:


mlp1=MLPClassifier(random_state=0)
mlp1.fit(x_train_scaled,y_train)
print(mlp1.score(x_train_scaled,y_train),":Training accuracy")
print(mlp1.score(x_test_scaled,y_test),":Test accuracy")

