#!/usr/bin/env python
# coding: utf-8

# # Goal
# 
# This notebook documents experiments on using different methods for feature selection.
# 
# We want to see how different algorithms perform feature selection vs. maintaining performance as the number of features increase.
# 
# We will have two types of **feature sets**:
# * Set 1: 2 types of features: informative + noise
# * Set 1: 4 two types of features: informative + noise + repeated (with some noise) + redundant (with some noise)
# and two types of **growth settings**:
# * Setting A: keep number of informative features fixed, grow noise
# * Setting B: grow informative features (and repeated/redundant) together with noise, keeping their proportions
# 
# 

# In[2]:


from boruta.boruta_py import BorutaPy
from sklearn.datasets import make_classification

#make_classification()


# In[3]:


def select_feature_boruta(X, y, 
                         perc=100,
                         alpha=0.05,
                         max_iter=100,
                         max_depth=7,
                         n_estimators='auto',
                         n_jobs=1):

    from sklearn.ensemble import RandomForestClassifier
    from boruta.boruta_py import BorutaPy
    
    X_is_df = isinstance(X, pd.DataFrame)
    y_is_df = isinstance(y, pd.Series)
    
    selector = BorutaPy(
        estimator=RandomForestClassifier(n_estimators=100, max_depth=max_depth, n_jobs=n_jobs),
        n_estimators=n_estimators,
        perc=perc,      
        alpha=alpha,    
        max_iter=max_iter,
        random_state=1,
        verbose=0,
    )

    # boruta needs a numpy array, not a dataframe
    X_train = X.values if X_is_df else X
    y_train = y.values if y_is_df else y

    selector.fit(X_train, y_train) 
    
    if X_is_df:
        columns = X.columns
        return sorted(np.array(columns)[selector.support_.tolist()])
    else:
        return sorted(selector.support_.tolist())


# In[ ]:




