#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time


# In[2]:


isAllNums = np.vectorize(lambda a, _: np.isreal(a))
isAnyNan = np.vectorize(lambda a, _: np.isnan(a))
isAnyInfinite = np.vectorize(lambda a, _: np.isinfinite(a))
fillNan = np.vectorize(lambda a, median: median if np.isnan(a) else a)
fillNanCat = np.vectorize(lambda a, _: str(a))


# In[3]:


train_data = pd.read_csv("hw3-data/my_train.csv")


# In[4]:


columns = list(train_data)


# In[5]:


feature_columns = columns[1:-1]


# In[6]:


np_train = np.array(train_data[feature_columns])


# In[7]:


np_train


# In[8]:


is_nums = isAllNums(np_train, -1)


# In[9]:


is_nums


# In[10]:


num_fields = []
for c in range(np_train.shape[1]):
    if np.all(is_nums[:,c]):
        num_fields.append(c);
        #np_train[:, c] = fillNan(np_train[:, c], np.median(np_train[:, c]))
        np_train[:, c] = fillNan(np_train[:, c], np.median(np_train[:, c]))
    else:
        np_train[:, c] = fillNanCat(np_train[:, c], -1)


# In[ ]:





# In[11]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(categories="auto", handle_unknown="ignore")
cat_1hot = cat_encoder.fit_transform(np_train)


# In[12]:


cat_1hot.toarray().shape


# In[13]:


np_train_labels = np.array(train_data["SalePrice"])


# In[14]:


np_train_labels


# In[15]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(cat_1hot, np_train_labels)


# In[16]:


dev_data = pd.read_csv("hw3-data/my_dev.csv")


# In[ ]:





# In[17]:


np_dev = np.array(dev_data)
np_dev = np_dev[:,1:-1]


# In[18]:


is_nums = isAllNums(np_dev, -1)
is_nums


# In[19]:


num_fields = []
for c in range(np_dev.shape[1]):
    if np.all(is_nums[:,c]):
        num_fields.append(c);
        median = np.median(np_dev[:, c])
        median = 0 #median if not np.isnan([median]) else 0
        np_dev[:, c] = fillNan(np_dev[:, c], median)
    else:
        np_dev[:, c] = fillNanCat(np_dev[:, c], -1)


# In[ ]:





# In[20]:


dev_1hot = cat_encoder.transform(np.array(np_dev))


# In[21]:


dev_1hot.shape


# In[22]:


p = lin_reg.predict(dev_1hot)


# In[23]:


l = np.array(dev_data)[:,-1]


# In[24]:


from sklearn.metrics import mean_squared_log_error
e = mean_squared_log_error(l, p)
np.sqrt(e)


# In[25]:


cats = []
for cat in cat_encoder.categories_:
    cats.append(len(cat))


# In[26]:


sum(cats)


# In[27]:


cats


# In[28]:


panda_cats = []
for fc in feature_columns:
    panda_cats.append(len(train_data[fc].unique()))


# In[41]:


panda_cats


# In[63]:


comparisons = []
indexes = []
for i, feature in enumerate(feature_columns):
    if cats[i] == panda_cats[i]:
        comparisons.append(cats[i])
    else:
        comparisons.append(feature)
        indexes.append(i);


# In[64]:


comparisons


# In[65]:


len(feature_columns)


# In[71]:


cat_encoder.categories_


# In[67]:


for fc in feature_columns:
    print(train_data[fc].unique())


# In[68]:


indexes


# In[72]:


cat_encoder.categories_[2]


# In[ ]:





# In[73]:


cat_encoder.categories_[25]


# In[ ]:





# In[77]:


cat_encoder.categories_[58]


# In[78]:


feature_columns


# In[81]:


len(lin_reg.coef_)


# In[82]:


lin_reg.intercept_


# In[30]:


train_data.describe()


# In[33]:


train_data.corr()["SalePrice"]


# In[ ]:




