
# coding: utf-8

# In[54]:

get_ipython().magic(u'config IPCompleter.greedy=True')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV


# In[4]:

from sklearn.datasets import load_digits
digits = load_digits()


# In[5]:

print("images shape: %s" % str(digits.images.shape))
print("targets shape: %s" % str(digits.target.shape))


# In[11]:

plt.matshow(digits.images[0], cmap=plt.cm.Greys);


# In[12]:

digits.target


# In[14]:

# prepare the data
X = digits.data.reshape(-1, 64)
print(X.shape)


# In[15]:

y = digits.target
print(y.shape)


# In[16]:

print(X)


# In[18]:

pca = PCA(n_components=2)


# In[19]:

pca.fit(X);


# In[20]:

X_pca = pca.transform(X)
X_pca.shape


# In[25]:

plt.figure(figsize=(8, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y);
plt.show()


# In[26]:

print(pca.mean_.shape)
print(pca.components_.shape)


# In[27]:

fix, ax = plt.subplots(1, 3)
ax[0].matshow(pca.mean_.reshape(8, 8), cmap=plt.cm.Greys)
ax[1].matshow(pca.components_[0, :].reshape(8, 8), cmap=plt.cm.Greys)
ax[2].matshow(pca.components_[1, :].reshape(8, 8), cmap=plt.cm.Greys);


# In[28]:

from sklearn.manifold import Isomap


# In[29]:

isomap = Isomap(n_components=2, n_neighbors=20)


# In[30]:

isomap.fit(X);


# In[31]:

X_isomap = isomap.transform(X)
X_isomap.shape


# In[33]:

plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=y);
plt.show()


# In[34]:

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[35]:

print("X_train shape: %s" % repr(X_train.shape))
print("y_train shape: %s" % repr(y_train.shape))
print("X_test shape: %s" % repr(X_test.shape))
print("y_test shape: %s" % repr(y_test.shape))


# In[38]:

svm = LinearSVC()


# In[39]:

svm.fit(X_train, y_train);


# In[40]:

svm.predict(X_train)


# In[41]:

svm.score(X_train, y_train)


# In[42]:

svm.score(X_test, y_test)


# In[46]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()


# In[47]:

rf.fit(X_train, y_train);


# In[48]:

rf.score(X_train, y_train)


# In[49]:

rf.score(X_test, y_test)


# In[52]:

from sklearn.cross_validation import cross_val_score
scores =  cross_val_score(rf, X_train, y_train, cv=5)
print("scores: %s  mean: %f  std: %f" % (str(scores), np.mean(scores), np.std(scores)))


# In[53]:

rf2 = RandomForestClassifier(n_estimators=50)
scores =  cross_val_score(rf2, X_train, y_train, cv=5)
print("scores: %s  mean: %f  std: %f" % (str(scores), np.mean(scores), np.std(scores)))


# In[57]:

param_grid = {'C': 10. ** np.arange(-3, 4)}
grid_search = GridSearchCV(svm, param_grid=param_grid, cv=3, verbose=3)


# In[58]:

grid_search.fit(X_train, y_train);


# In[59]:

print(grid_search.best_params_)
print(grid_search.best_score_)


# In[60]:

print(grid_search.best_params_)
print(grid_search.best_score_)


# In[ ]:



