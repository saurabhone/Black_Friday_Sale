#!/usr/bin/env python
# coding: utf-8

# # BLACK FRIDAY SALE

# #### I have collected the purchase data of a superstore on the Black Friday Sale. I will analyse the data for better inventory management and increasing the sale for the future Black Friday Sale.
# #### We will use Python libraries such as Numpy, Pandas, Scipy, Matplotlib, Seaborn, Plotly, Scikit-Learn, etc for achieving our goal.

# ## IMPORT LIBRARIES

# In[1]:


# We will import all packages before we import our data set

get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import pyplot
import scipy.stats as stats
from scipy.stats import chisquare 
sns.set(style='ticks', context='talk') 
import plotly.plotly as py
import plotly 
plotly.tools.set_credentials_file(username='saurabhone', api_key='D1gcJm0ztIvbBba8OMl2')


# ## IMPORT DATA

# In[ ]:


# reading data set using pandas function

d = pd.read_csv('/Users/saurabhkarambalkar/Desktop/bb/data.csv')
d.head()


# ## CLEANING DATA

# In[2]:


d.info()


# #### We see that there are missing values in Electronics and Furniture which means that people did not buy from these two departments and thus we need to fill them with the value zero.

# In[3]:


d.fillna(value=0,inplace=True)
d.head()


# In[4]:


d.info()


# #### Now that we see there are no missing values in our data, we can start with exploration part.

# ## EXPLORATORY DATA ANALYSIS

# In[5]:


sns.distplot(d['Purchase'])


# #### The purchase trend graph gives us an insight that most of the people purchased within the amount 5000 and 10000

# In[6]:


# Countplot of Male and female(to compare the total count and the purchase done by the higher and the lower gender)
sns.countplot(x ='Gender', data = d)


# In[8]:


#barplot for categorical vs numerical
sns.set_style('ticks')
sns.barplot(x='Gender',y='Purchase',data = d)
sns.despine(left=True,bottom=True)


# #### From the above two graphs we observe that there were more number of Males than Females who did the shopping. Also, even though the count of Females were low, the total amount spent by the Females is much closer to Males. 

# In[9]:


#Boxplot is used to check the purchase range by different age groups
sns.boxplot(x='Age',y='Purchase',data =d)


# #### The age group 51-55 spent the most amount in shopping. 

# In[10]:


#Boxplot is used to check the purchase range by different age groups (which are further segregated by Marital status)
fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(11, 8)    
sns.boxplot(x='Age',y='Purchase',data =d,hue='Marital_Status', ax=ax)


# #### From the above graph it is clearly understood that the unmarried customers spent the most amount than the married customers.

# In[11]:


sns.violinplot(x = 'State', y = 'Purchase', data = d, hue = 'Gender', split = True)


# In[12]:


sns.factorplot(x = 'State', y = 'Purchase', data = d, kind = 'bar')


# #### The bar graph shows that the State which spent the most is PA and the violin plot shows that the most purchases were made by PA Males. 

# In[13]:


tc = d.corr()
sns.heatmap(tc)


# #### From the correlation graph we observe that the Furniture is more correlated to the Purchase.

# #### We will plot some interesting interactive plotly graphs for insights

# In[14]:


from plotly import __version__
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
plotly.offline.init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()


# In[15]:


df = pd.DataFrame(data = d, columns = ['Apparels','Purchase'])
df.iplot()


# In[18]:


d.iplot(kind = 'surface', colorscale='rdylbu')


# In[19]:


df1 = pd.DataFrame(data = d, columns = ['State','Gender','Purchase'])
df1.iplot()


# In[20]:


df2 = pd.DataFrame(data = d, columns = ['State','Gender','Apparels','Purchase'])


# In[21]:


df2.iplot()


# In[22]:


#We use pd.to_numeric function to change the data type to integer

d.Electronics = pd.to_numeric(d.Electronics, errors='coerce')
d = d.dropna(subset=['Electronics'])
d.Electronics = d.Electronics.astype(int)

d.Furniture = pd.to_numeric(d.Furniture, errors='coerce')
d = d.dropna(subset=['Furniture'])
d.Furniture = d.Furniture.astype(int)


# In[23]:


final= d[['State', 'Apparels', 'Electronics','Furniture','Purchase']].copy()
final[:5]


# In[24]:


NYC= final[(final['State'] == 'NYC')]
print('Apparels:',sum(NYC['Apparels']))
print('Electronics:',sum(NYC['Electronics']))
print('Furniture:',sum(NYC['Furniture']))
NYC_apparels=sum(NYC['Apparels'])
NYC_electronics=sum(NYC['Electronics'])
NYC_furniture=sum(NYC['Furniture'])
NYC_Purchase=NYC_apparels+NYC_electronics+NYC_furniture


# In[25]:


NJ= final[(final['State'] == 'NJ')] 
print('Apparels:',sum(NJ['Apparels']))
print('Electronics:',sum(NJ['Electronics']))
print('Furniture:',sum(NJ['Furniture']))
NJ_apparels=sum(NJ['Apparels'])
NJ_electronics=sum(NJ['Electronics'])
NJ_furniture=sum(NJ['Furniture'])
NJ_Purchase=NJ_apparels+NJ_electronics+NJ_furniture


# In[26]:


PA= final[(final['State'] == 'PA')] 
print('Apparels:',sum(PA['Apparels']))
print('Electronics:',sum(PA['Electronics']))
print('Furniture:',sum(PA['Furniture']))
PA_apparels=sum(PA['Apparels'])
PA_electronics=sum(PA['Electronics'])
PA_furniture=sum(PA['Furniture'])
PA_Purchase=PA_apparels+PA_electronics+PA_furniture


# In[27]:


f = {'State':['NY','NJ','PA'],
     'Apparels': [NYC_apparels,NJ_apparels,PA_apparels], 
     'Electronics': [NYC_electronics, NJ_electronics,PA_electronics],  
     'Furniture': [NYC_furniture,NJ_furniture,PA_furniture],
     'Purchase':[NYC_Purchase,NJ_Purchase,PA_Purchase]
    }
f1 = pd.DataFrame(data=f)
print(f1)


# In[28]:



for col in f1.columns:
    f1[col] = f1[col].astype(str)

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

f1['text'] = f1['State'] + '<br>' +    'Apparels '+f1['Apparels']+' Electronics '+f1['Electronics']+'<br>'+    'Furniture '+f1['Furniture']
    

    
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = f['State'],
        z = f1['Purchase'].astype(float),
        locationmode = 'USA-states',
        text = f1['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "USD")
        ) ]

layout = dict(
        title = '2017 Black Friday Sale in Tri State Region<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename='d3-cloropleth-map') 


# ## PREDICTION MODELING

# #### Now using different features we will predict the Purchase which can help us in the future Black Friday Sales

# In[54]:


#Selecting the features for prediction

data = d[['Gender','Apparels', 'Furniture', 'Electronics','Purchase']]
X = data.iloc[:,:-1].values
y = data.iloc[:,4].values


# In[55]:


#Encoding the categorical variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()


# In[56]:


#Adding the dummy variable trap

X = X[:,1:]


# In[57]:


#Spliting the data into train and test sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[58]:


#Fitting Multiple Linear Regression Model on train set 

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)


# In[60]:


#Predicting the test set results

y_pred = reg.predict(X_test)
y_pred


# In[92]:


#Building the optimum model using Backward Elimination Technique

import statsmodels.formula.api as sm
X=np.append(arr = np.ones((65499,1)).astype(int),values=X,axis=1)
X_opt = X[:,[0,1,2,3,4]]
reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
reg_OLS.summary()


# In[94]:


#The p value of Gender is more than our significance value of 0.05 (5%) so we need eliminate it from our model

X_opt = X[:,[0,2,3,4]]
reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
reg_OLS.summary()


# ### The model we have obtained is best suitable for prediction of Purchase for future Black Friday Sales
