#!/usr/bin/env python
# coding: utf-8

# In[94]:


import pandas as pd
import numpy as np
import time


# In[95]:


isAllNums = np.vectorize(lambda a, _: np.isreal(a))
isAnyNan = np.vectorize(lambda a, _: np.isnan(a))
isAnyInfinite = np.vectorize(lambda a, _: np.isinfinite(a))
fillNan = np.vectorize(lambda a, median: median if np.isnan(a) else a)
fillNanCat = np.vectorize(lambda a, _: str(a))


# In[96]:


train_data = pd.read_csv("hw3-data/my_train.csv")


# In[97]:


columns = list(train_data)


# In[98]:


feature_columns = columns[1:-1]


# In[99]:


np_train = np.array(train_data[feature_columns])


# In[100]:


is_nums = isAllNums(np_train, -1)


# In[101]:


num_fields = []
for c in range(np_train.shape[1]):
    if np.all(is_nums[:,c]):
        num_fields.append(c);
        #np_train[:, c] = fillNan(np_train[:, c], np.median(np_train[:, c]))
        np_train[:, c] = fillNan(np_train[:, c], np.median(np_train[:, c]))
    else:
        np_train[:, c] = fillNanCat(np_train[:, c], -1)


# In[102]:


train_nums = np_train[:, num_fields]


# In[103]:


train_cat_fields = [i for i, x in enumerate(feature_columns) if i not in num_fields]


# In[104]:


train_cats = np_train[:, train_cat_fields]

train_cat_fields
# In[105]:


train_cats


# In[106]:


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(categories="auto", handle_unknown="ignore")
cat_1hot = cat_encoder.fit_transform(train_cats)


# In[107]:


smart_train = np.append(train_nums, cat_1hot.toarray(), axis=1)


# In[108]:


train_nums.shape


# In[109]:


cat_1hot.toarray().shape


# In[110]:


smart_train


# In[111]:


smart_train.shape


# In[112]:


np_train_labels = np.array(train_data["SalePrice"])


# In[113]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(smart_train, np_train_labels)


# In[114]:


dev_data = pd.read_csv("hw3-data/my_dev.csv")


# In[115]:


np_dev = np.array(dev_data)
np_dev = np_dev[:,1:-1]


# In[116]:


dev_nums = np_dev[:, num_fields]
dev_cats = np_dev[:, train_cat_fields]


# In[117]:


num_fields = []
for c in range(np_dev.shape[1]):
    if np.all(is_nums[:,c]):
        num_fields.append(c);
        median = np.median(np_dev[:, c])
        median = 0 #median if not np.isnan([median]) else 0
        np_dev[:, c] = fillNan(np_dev[:, c], median)
    else:
        np_dev[:, c] = fillNanCat(np_dev[:, c], -1)


# In[118]:


dev_nums = fillNan(dev_nums, median)


# In[119]:


dev_cats = fillNanCat(dev_cats, -1)


# In[120]:


dev_1hot = cat_encoder.transform(dev_cats)


# In[121]:


smart_dev = np.append(dev_nums, dev_1hot.toarray(), axis=1)


# In[122]:


p = lin_reg.predict(smart_dev)


# In[123]:


l = np.array(dev_data)[:,-1]


# In[124]:


from sklearn.metrics import mean_squared_log_error
e = mean_squared_log_error(l, p)
np.sqrt(e)


# In[125]:


lin_reg.intercept_


# In[126]:


dev_nums


# In[ ]:





# In[127]:


len(l)


# In[128]:


len(p)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




