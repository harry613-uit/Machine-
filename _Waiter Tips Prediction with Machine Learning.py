#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv("tips.csv")
print(data.head())


# In[19]:


figure = px.pie(data, 
            values='tip', 
            names='sex',hole = 0.5)
figure.show()


# In[25]:


figure = px.pie(data, 
             values='tip', 
             names='smoker',hole = 0.5)
figure.show()


# In[26]:


figure = px.pie(data, 
             values='tip', 
             names='time',hole = 0.5)
figure.show()


# In[27]:


data["sex"] = data["sex"].map({"Female": 0, "Male": 1})
data["smoker"] = data["smoker"].map({"No": 0, "Yes": 1})
data["day"] = data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
data["time"] = data["time"].map({"Lunch": 0, "Dinner": 1})
data.head()


# In[33]:


x = np.array(data[["total_bill", "sex", "smoker", "day", 
                   "time", "size"]])
y = np.array(data["tip"])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                test_size=0.2, 
                                                random_state=42)


# In[34]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)


# In[37]:


figure = px.scatter(data_frame = data, x="total_bill",
                    y="tip", size="size", color= "time", trendline="ols")
figure.show()


# In[38]:


figure = px.scatter(data_frame = data, x="total_bill",
                    y="tip", size="size", color= "day", trendline="ols")
figure.show()


# In[ ]:




