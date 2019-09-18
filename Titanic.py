

#import packages
import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns

train=pandas.read_csv(r"F:\ml material\Dataset\titanic\titanictrain.csv")
test=pandas.read_csv(r"F:\ml material\Dataset\titanic\titanictest.csv")

train.info()
test.info()

##########Feature engineering and cleaning
data=pandas.concat([train.drop(['Survived'],axis=1),test])
data.shape
data.info()

#check for missing values
data.isnull().sum()

data['HasCabin']= ~data['Cabin'].isnull()
data.isnull().sum()

#cleaning the age column
data['Age']=data['Age'].fillna(data['Age'].median())
data['Fare']=data['Fare'].fillna(data['Fare'].median())

data['Embarked']=data['Embarked'].fillna('S')


data.drop(['Name','Ticket','Cabin','PassengerId'],axis=1,inplace=True)

plt.figure(figsize=(12,5))
sns.countplot(train['Pclass'])
plt.show()
plt.figure(figsize=(12,5))
sns.countplot(train['Pclass'][train['Survived']==1])
plt.show()

plt.figure(figsize=(12,5))
sns.countplot(train['Sex'],order=['male','female'])
plt.show()
plt.figure(figsize=(12,5))
sns.countplot(train['Sex'][train['Survived']==1],order=['male','female'])
plt.show()

plt.figure(figsize=(12,5))
sns.countplot(train['SibSp'])
plt.show()
plt.figure(figsize=(12,5))
sns.countplot(train['SibSp'][train['Survived']==1])
plt.show()


plt.figure(figsize=(12,5))
sns.countplot(train['Parch'])
plt.show()
plt.figure(figsize=(12,5))
sns.countplot(train['Parch'][train['Survived']==1])
plt.show()


data['family']=data['SibSp']+data['Parch']

plt.figure(figsize=(12,5))
sns.countplot(train['Embarked'],order=['S','C','Q'])
plt.show()
plt.figure(figsize=(12,5))
sns.countplot(train['Embarked'][train['Survived']==1],order=['S','C','Q'])
plt.show()

data.drop(['SibSp','Parch'],axis=1,inplace=True)
traind=data.iloc[:891,:]
test=data.iloc[891:,:]
traind['Survived']=train['Survived']


plt.figure(figsize=(12,5))
sns.distplot(traind['Age'][traind['Survived']==1])
sns.distplot(traind['Age'][traind['Survived']==0])
plt.legend(['1','0'])
plt.show()

plt.figure(figsize=(12,5))
sns.distplot(traind['Fare'][traind['Survived']==1])
sns.distplot(traind['Fare'][traind['Survived']==0])
plt.legend(['1','0'])
plt.show()

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le1=LabelEncoder()
traind['Sex']=le1.fit_transform(traind['Sex'])

le2=LabelEncoder()
traind['Embarked']=le2.fit_transform(traind['Embarked'])

xdata=traind.drop(['Survived'],axis=1)
ydata=traind['Survived']

ohe=OneHotEncoder(categorical_features=[4])
xdata=ohe.fit_transform(xdata).toarray()

from sklearn.tree import DecisionTreeClassifier
alg=DecisionTreeClassifier(max_depth=10)

alg.fit(xdata,ydata)


#prediction
test['Sex']=le1.fit_transform(test['Sex'])
test['Embarked']=le2.fit_transform(test['Embarked'])
test=ohe.transform(test).toarray()
ypred=alg.predict(test)



prediction=pandas.read_csv(r"F:\ml material\Dataset\titanic\titanictest.csv")

prediction['Survived']=ypred

prediction=prediction[['PassengerId','Survived']]
prediction.to_csv('prediction.csv')





