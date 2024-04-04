#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
data = pd.read_csv(r'C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\olympics.csv')
print(data)


# In[3]:


import pandas as pd
regions = pd.read_csv(r'C:\ProgramData\MySQL\MySQL Server 8.0\Uploads\noc_regions.csv')
print(regions)#national olympics committee


# # Age of players

# In[4]:


first = data["Age"] #age of all players
print(first)


# # details of first 5 rows

# In[5]:


print(data.head(5)) #details of first 5 players


# # describe data

# In[6]:


print(data.head(), data.describe(), data.info())


# # info of data

# In[7]:


print(data.info())


# # merging 2 tables

# In[8]:


merged = pd.merge(data, regions, on ='NOC', how='left')
print(merged.head())


# # players who won gold medals

# In[9]:


goldMedals = merged[(merged.Medal== 'Gold')] #who win gold medal
print(goldMedals.head())


# # womens involve in olympics

# In[10]:


womens = merged[(merged.Sex == 'F')] #womens involved in olympics
print(womens)


# In[11]:


import pylab as plt
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(16,12)})


# # count of Teams & NOC

# In[12]:


print("Number of Nationalities -- >", len(data.NOC.unique()))
print("Number of Teams -->", len(data.Team.unique()))


# # checking sum of null values

# In[13]:


data.isnull().sum()


# # total count of Medal distribution

# In[14]:


data.Medal.value_counts()


# # Summer olympics conducted years

# In[15]:


print("Summer olympics conduct in", np.array(sorted(data[data['Season'] == 'Summer']['Year'].unique())))


# # Winter olympics conducted years

# In[16]:


print("Winter olympics conduct in", np.array(sorted(data[data['Season'] == 'Winter']['Year'].unique())))


# # Number of players participated in each olympics

# In[17]:


groupedYearID = data.groupby(['Year','ID'], as_index= False).count()[['Year','ID']]
groupedYearID = groupedYearID.groupby('Year', as_index=False).count()
groupedYearID.head(10)


# # creating seperate columns for meadls 

# In[18]:


data = pd.concat([data,pd.get_dummies(data.Medal)],axis=1)
data['allmedals']= data['allmedals']= data['Bronze'] + data['Gold'] + data['Silver']
data.head()


# # NOC who won most medals

# In[19]:


groupcountry = data.groupby(by=['NOC'], as_index=False).sum()
top50 = groupcountry.sort_values(by=['allmedals'], ascending = False).head(50)
plot2 = sns.barplot('NOC','allmedals', data=top50).set_xticklabels(top50.NOC,rotation=82)


# In[ ]:





# # summer rank & winter rank

# In[20]:


x = np.array(['summer_rank'])         
y = np.array(['winter_rank'])

plt.scatter(x,y,color = 'red')
plt.show()
#(data_frame = data, x = 'summer_rank', y = 'winter_rank', color = 'red', size = 'overall_rank',
         # labels = {'summer_rank': 'Summer Rank', 'winter_rank': 'Winter Rank'}, title = 'Summer Rank vs. Winter Rank')


# In[21]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#import tensorflow as tf
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# # summary of season cloumn

# In[22]:


# seeing group summary of season cloumn
data['Season'].value_counts()


# # summary of Medal cloumn

# In[23]:


# seeing group summary of Medal cloumn
data['Medal'].value_counts()


# # summary of Event cloumn

# In[24]:


# seeing group summary of Event cloumn
data['Event'].value_counts()


# # summary of Team cloumn

# In[25]:


# seeing group summary of Team cloumn
data['Team'].value_counts()


# # Exploratory Data Analysis

# # Age Distribution

# In[26]:


# Age Distribution

plt.figure(figsize=(15,7))
plt.title("Age VS No. of Participants")
plt.hist(data.Age,color='purple', bins = 35)
plt.xlabel("Age")
plt.ylabel("Participants")
plt.show()


# In[27]:


# Age to participant distribution
age_dist = data.Age.value_counts().sort_values(ascending=False).head(15)
age_dist


# # Gender Distribution

# In[28]:


# Gender Distribution
gender_unique_values = data.Sex.value_counts()
gender_unique_values


# In[29]:


plt.figure(figsize=(15,7))
plt.title("Gender Distribution")
plt.pie(gender_unique_values, labels=gender_unique_values.index, autopct="%.2f%%", startangle=90)
plt.show()


# # female participants in each year in summer season

# In[30]:


# seeing female participants in each year in summer season

female_part = data[(data.Sex=='F') & (data.Season == 'Summer')][['Sex', 'Year']]
female_part = female_part.groupby('Year').count().reset_index()
female_part


# # Visualizing the female participant data

# In[31]:


# Visualizing the female participant data

plt.figure(figsize=(15,7))
sns.lineplot(data=female_part, x='Year', y='Sex', linestyle = '--', color = 'purple')
plt.xlabel('Year')
plt.ylabel('Female Participants')


# # Participants across the season

# In[32]:


# Participants across the season

seasons = data.Season.value_counts()
seasons


# # Visualizing paritipant distribution in different season

# In[33]:


# Visualizing paritipant distribution in different season
fig1, ax1 = plt.subplots()

colors = ['#99ff96','#000000']

ax1.pie(seasons, colors = colors, labels=seasons.index, autopct='%1.1f%%')

centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

ax1.axis('equal')  
plt.tight_layout()
plt.show()


# # countries wise visualization

# # Top 15 Countries participating

# In[34]:


# Top 15 Countries participating in Olympics by no. of participants
top_countries = data.Team.value_counts().sort_values(ascending=False).head(15)
top_countries
     


# In[35]:


# Data Visualization of countries participating
sns.set_style('darkgrid')
plt.figure(figsize=(15, 7))
plt.title('Top 15 Countries Total Participations')
plt.xticks(rotation=75)
sns.barplot(x=top_countries.index, y=top_countries)


# # Data of countries with most Gold medals

# In[36]:


# Data of countries with most Gold medals

max_year_country = data[(data.Medal == 'Gold')].Team
max_year_country = max_year_country.value_counts().head(20)
max_year_country


# In[37]:


# Visulaizing the countries with most Gold medals
plt.figure(figsize=(15,7))
sns.barplot(x=max_year_country, y=max_year_country.index)
plt.xlabel("Top 20 Gold Medal Won Countrywise")


# # Age, Height & Weight wise data Visualization

# In[38]:


# Age vs Height Spread

age_heig = data[(data.Age != 0) & (data.Height != 0.0) & (data.Medal != 'None') & (data.Season == 'Summer')]
plt.figure(figsize=(15,7))
sns.scatterplot(x=age_heig.Age, y=age_heig.Height, data=age_heig, hue='Sex')
plt.xlabel('Age')
plt.ylabel('Height')
     


# In[39]:


# Age vs Weight Spread

age_weig = data[(data.Age != 0.0) & (data.Weight != 0.0) & (data.Medal != 'None') & (data.Season == 'Summer')]
plt.figure(figsize=(15,7))
sns.scatterplot(x=age_weig.Age, y=age_weig.Weight, data=age_weig, hue='Sex')
plt.xlabel('Age')
plt.ylabel('Weight')


# In[40]:


# Weight & Height Spread

heig_weight = data[(data.Height != 0.0) & (data.Age != 0) & (data.Weight != 0.0) & (data.Medal != 'None') & (data.Season == 'Summer')]
plt.figure(figsize=(15,7))
sns.scatterplot(x=heig_weight.Weight, y=heig_weight.Height, data=heig_weight, hue='Sex')
plt.xlabel('Weight')
plt.ylabel('Height')


# # Data Cleaning

# In[41]:


# To fill missing values in Medal column with 0's and 1's
data['Medal'] = data['Medal'].apply(lambda x: 1 if str(x) != 'nan' else 0)


# In[42]:


data


# In[43]:


# Drop Uncessary feature columns
data = data.drop(['ID', 'Name', 'Games'], axis=1)
     


# In[44]:


# Checking null values in the data 
data.isna().mean()


# In[45]:


data.groupby(['Medal', 'Sex']).mean().astype(np.int)


# In[46]:


# Fill null values with mean values for these columns
for column in ['Age', 'Height', 'Weight']:
    data[column] = data.groupby(['Medal', 'Sex'])[column].apply(lambda x: x.fillna(x.mean()).astype(np.int))


# In[47]:


# Checking null values again
print("Total missing values:", data.isna().sum().sum())


# In[48]:


data


# In[49]:


# Checking no. of unique values in the column
{column: len(data[column].unique()) for column in data.select_dtypes('object').columns}


# In[50]:


# defining the function.
def binary_encode(df, columns, positive_values):
    df = df.copy()
    for column, positive_value in zip(columns, positive_values):
        df[column] = df[column].apply(lambda x: 1 if x == positive_value else 0)
    return df

def onehot_encode(df, columns, prefixes):
    df = df.copy()
    for column, prefix in zip(columns, prefixes):
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df


# In[51]:


data = binary_encode(
    data,
    columns=['Sex', 'Season'],
    positive_values=['M', 'Summer']
)

data = onehot_encode(
    data,
    columns=['Team', 'NOC', 'City', 'Sport', 'Event'],
    prefixes=['T', 'N', 'C', 'S', 'E']
)


# In[52]:


data


# In[53]:


# Spilt data in target column and features columns
y = data['Medal']
X = data.drop('Medal', axis=1)


# # Feature Selection

# In[54]:


#Apply SelectKBest and extract top 10 features out of the data
best = SelectKBest(score_func=chi2, k=10)


# In[55]:


fit = best.fit(X,y)


# In[56]:


data_scores=pd.DataFrame(fit.scores_)
data_columns=pd.DataFrame(X.columns)


# In[57]:


# Join the two dataframes
scores=pd.concat([data_columns,data_scores],axis=1)
scores.columns=['Feature','Score']
print(scores.nlargest(11,'Score'))


# In[58]:


# Select 10 features
features=scores["Feature"].tolist()[:10]
features


# # Make new dataset with cleaned data

# In[59]:


data=data[['Age','Sex','Height','Weight','Year','Season','T_Afghanistan','T_Algeria','T_Angola','T_Argentina','Medal']]
data.head()


# In[60]:


y = data['Medal']
X = data.drop(['Medal'], axis=1)

#Split data into training and testing data
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.4, random_state=1)


# # Feature Scaling

# In[61]:


# Scaling data 
scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)


# # Machine Learning

# # 1. Logistic Regression

# In[62]:


# 1. Logistic Regression

algo1 = 'LogisticRegression'
lr = LogisticRegression(random_state=1, max_iter=1000)
lr.fit(train_x, train_y)
lr_predict = lr.predict(test_x)
lr_conf_matrix = confusion_matrix(test_y, lr_predict)
lr_acc_score = accuracy_score(test_y, lr_predict)
print("confusion matrix")
print(lr_conf_matrix)
print("\n")
print("Accuracy of Logistic Regression:",lr_acc_score*100,'\n')
print(classification_report(test_y,lr_predict))


# # 2. Multinomial Naive Bayes

# In[63]:


# 2. Multinomial Naive Bayes

algo2 = 'MultinomialNB'
nv = MultinomialNB()
nv.fit(train_x, train_y)
nv_predict = nv.predict(test_x)
nv_conf_matrix = confusion_matrix(test_y, nv_predict)
nv_acc_score = accuracy_score(test_y, nv_predict)
print("confusion matrix")
print(nv_conf_matrix)
print("\n")
print("Accuracy of Logistic Regression:",nv_acc_score*100,'\n')
print(classification_report(test_y,nv_predict))


# # 3. Decision Tree 
# 

# In[64]:


# 3. Decision Tree 

algo3 = 'DecisionTreeClassifier'
dt = DecisionTreeClassifier(criterion = 'entropy',random_state=10,max_depth = 30)
dt.fit(train_x,train_y)
dt_predicted = dt.predict(test_x)
dt_conf_matrix = confusion_matrix(test_y, dt_predicted)
dt_acc_score = accuracy_score(test_y, dt_predicted)
print("confusion matrix")
print(dt_conf_matrix)
print("\n")
print("Accuracy of DecisionTreeClassifier:",dt_acc_score*100,'\n')
print(classification_report(test_y,dt_predicted))


# # 4. Random Forest

# In[65]:


# 4. Random Forest 

algo4 = 'Random Forest Classfier'
rf = RandomForestClassifier(n_estimators=200, random_state=10,max_depth=15)
rf.fit(train_x,train_y)
rf_predicted = rf.predict(test_x)
rf_conf_matrix = confusion_matrix(test_y, rf_predicted)
rf_acc_score = accuracy_score(test_y, rf_predicted)
print("confusion matrix")
print(rf_conf_matrix)
print("\n")
print("Accuracy of Random Forest:",rf_acc_score*100,'\n')
print(classification_report(test_y,rf_predicted))


# # 5. Gradient Boosting

# In[66]:


# 5. Gradient Boosting 

algo5 = 'Gradient Boosting Classifier'
gvc =  GradientBoostingClassifier()
gvc.fit(train_x,train_y)
gvc_predicted = gvc.predict(test_x)
gvc_conf_matrix = confusion_matrix(test_y, gvc_predicted)
gvc_acc_score = accuracy_score(test_y, gvc_predicted)
print("confusion matrix")
print(gvc_conf_matrix)
print("\n")
print("Accuracy of Gradient Boosting Classifier:",gvc_acc_score*100,'\n')
print(classification_report(test_y,gvc_predicted))


# # Model Evaluation

# In[67]:


#Evaluating all the Algorithms at once.
model_ev = pd.DataFrame({'Model': ['Logistic Regression','MultinomialNB','Decision Tree','Random Forest',
                                  'Gradient Boosting'], 
                         'Accuracy': [lr_acc_score*100, nv_acc_score*100, dt_acc_score*100, rf_acc_score*100,
                                      gvc_acc_score*100]})
model_ev


# In[ ]:





# In[ ]:




