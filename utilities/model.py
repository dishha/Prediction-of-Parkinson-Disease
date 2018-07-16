
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import pickle


# In[2]:


df = pd.read_csv("C:\Users\user\Desktop\disha\mlModel\df.csv")


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:22], df.iloc[:,23], test_size = 0.3, random_state = 100)


# In[4]:


clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)


# In[5]:


clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)


# In[6]:


y_pred = clf_gini.predict(X_test)
y_pred


# In[7]:


y_pred_en = clf_entropy.predict(X_test)
y_pred_en


# In[8]:


print "Accuracy is ", accuracy_score(y_test,y_pred)*100


# In[9]:


print "Accuracy is ", accuracy_score(y_test,y_pred_en)*100


# In[10]:


from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

pipeline = Pipeline([
    ('clf', DecisionTreeClassifier(criterion='entropy'))
])

parameters = {
    'clf__max_depth': (10,15, 20),
    'clf__min_samples_split':(1.0, 2, 3),
    'clf__min_samples_leaf': (1, 2, 3)
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,
                           verbose=1, scoring= 'f1' )
grid_search.fit(X_train, y_train)
print 'Best score: %0.3f' % grid_search.best_score_
predicts = grid_search.predict(X_test)
print classification_report(y_test, predicts)


# In[56]:


from sklearn.ensemble import RandomForestClassifier


pipeline = Pipeline([
    ('clf',RandomForestClassifier(criterion='entropy'))
])

parameters = {
    'clf__n_estimators': (5, 10,20,50),
    'clf__max_depth': (10,15,20),
    'clf__min_samples_split':(1.0, 2, 3),
    'clf__min_samples_leaf': (1, 2, 3)
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1,
                           verbose=1, scoring= 'f1' )
grid_search.fit(X_train, y_train)
print 'Best score: %0.3f' % grid_search.best_score_
predicts = grid_search.predict(X_test)
print classification_report(y_test, predicts)


# In[12]:


filename = 'mlModel.sav'
pickle.dump(grid_search,open(filename,'wb'))


# In[13]:


loaded_model = pickle.load(open(filename,'rb'))
result = loaded_model.score(X_test,y_test)
print(result)


# In[14]:


list_pickle = pickle.dumps(result)


# In[15]:


list_pickle


# In[16]:


loaded_pickle = pickle.loads(list_pickle)


# In[17]:


loaded_pickle


# In[18]:


df.dtypes


# In[24]:


pscore = metrics.accuracy_score(y_test, predicts)


# In[25]:


print pscore


# In[36]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = pd.DataFrame(confusion_matrix(y_test, predicts), columns= [0,1], index=[0,1])
sns.heatmap(cm, annot=True)


# In[38]:


cm


# In[45]:


TP = cm[1][1]
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]


# In[46]:


print((TP + TN) / float(TP + TN + FP + FN))
print(metrics.accuracy_score(y_test, predicts))


# In[48]:


classification_error = (FP + FN) / float(TP + TN + FP + FN)

print(classification_error)
print(1 - metrics.accuracy_score(y_test, predicts))


# In[49]:


sensitivity = TP / float(FN + TP)

print(sensitivity)
print(metrics.recall_score(y_test, predicts))


# In[50]:


specificity = TN / (TN + FP)

print(specificity)


# In[51]:


false_positive_rate = FP / float(TN + FP)

print(false_positive_rate)
print(1 - specificity)


# In[53]:


precision = TP / float(TP + FP)

print(precision)
print(metrics.precision_score(y_test, predicts))


# In[54]:


from sklearn.metrics import roc_auc_score

p = t1.predict_proba(X_test)[:, 1]


# In[58]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
roc_auc = roc_auc_score(y_test, grid_search.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, grid_search.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

