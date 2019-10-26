#!/usr/bin/env python
# coding: utf-8

# Random forest is connsidered to be best and one of the best and mosstly used for supervised learning algorithm
# Tree Based models work well on non linear mapping
# and can work on both classfication and regression problems
# 
# 
# 
# 

# Most of the times it is used for feature selection
# We will use iris flower dataset for demonstration purpose

# In[ ]:


from sklearn import datasets
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[19]:


data = datasets.load_iris()
data.data.shape


# In[20]:


data.target_names


# In[21]:


data.feature_names


# In[23]:


data = pd.DataFrame({
    'sepal length': data.data[:,0],
    'sepal width': data.data[:, 1],
    'petal length': data.data[:, 2], 
    'petal width': data.data[:, 3],
    'species': data.target
})


# In[25]:


data.head()


# In[26]:


X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]
Y = data['species']


# In[27]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =0.3)


# In[31]:


model  = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)
predicted = model.predict(X_test)


# In[33]:


print("Accuracy: ", accuracy_score(Y_test, predicted))


# In[34]:


model.predict([[2,5,1,3]])


# In[53]:


# Second part finding important features

from sklearn.ensemble import RandomForestClassifier

model1 = RandomForestClassifier(n_estimators=100)
model1.fit(X_train, Y_train)
data = datasets.load_iris()


# In[54]:


imp_feature = pd.Series(model1.feature_importances_, index=data.feature_names).sort_values(ascending=False)


# In[55]:


imp_feature


# In[56]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.barplot(x=imp_feature, y=imp_feature.index)
plt.xlabel('Feature Importance Score')
plt.ylabel("Features")
plt.title("Visualizing Important features")
plt.legend()
plt.show()


# In[51]:


data.feature_names


# In[57]:


from sklearn.model_selection import train_test_split

data= pd.DataFrame({
    'sepal length': data.data[:, 0],
    'petal length': data.data[:, 2], 
    'petal width': data.data[:, 3],
    'species': data.target,
})


# In[58]:


data.head(3)


# In[71]:


X = data[['petal length', 'petal width', 'sepal length']]
Y = data['species']


# In[72]:


X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size=0.7, random_state=5)
Y_train


# In[75]:


from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier(n_estimators=100)

model2


# In[ ]:


model2.fit(X_train, Y_train)
pred2 =model2.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, pred2))

