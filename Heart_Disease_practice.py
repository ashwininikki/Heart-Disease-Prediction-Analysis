#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Basic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[4]:


# Other libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[5]:


# Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[6]:


import pandas as pd
df = pd.read_csv(r'C:/Users/Ashwini Yadav/HeartDisease.csv')
df.info()


# In[18]:


df.describe()


# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[25]:


df.hist()


# In[26]:


sns.set_style('whitegrid')
sns.countplot(x='target',data=df,palette='RdBu_r')


# In[34]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
col_names = ['age', 'gender', 'chest_pain', 'rest_bps', 'cholestrol', 'fasting_blood_sugar', 'rest_ecg', 'thalach','excer_angina','old_peak','slope','ca','thalassemia','target']
pima = pd.read_csv(r"C:/Users/Ashwini Yadav/HeartDisease.csv", header=None, names=col_names)
pima.head()


# In[17]:


import pandas as pd
df = pd.read_csv(r'C:/Users/Ashwini Yadav/HeartDisease.csv')

y = df['target']
X = df.drop(['target'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

dt_scores = []
for i in range(1, len(X.columns) + 1):
    dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
    dt_classifier.fit(X_train, y_train)
    dt_scores.append(dt_classifier.score(X_test, y_test))
    
plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green')
for i in range(1, len(X.columns) + 1):
    plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))
plt.xticks([i for i in range(1, len(X.columns) + 1)])
plt.xlabel('Max features')
plt.ylabel('Scores')
plt.title('Decision Tree Classifier scores for different number of maximum features')


# In[ ]:




