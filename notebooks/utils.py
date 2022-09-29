import numpy as np
import shap

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

class SHAPImportanceRandomForestClassifier(RandomForestClassifier):
    def fit(self, X, y, sample_weight=None):
        self.X_ = X
        super().fit(X, y, sample_weight=sample_weight)
        return self
    @property
    def feature_importances_(self):
        check_is_fitted(self)
        explainer = shap.TreeExplainer(self)
        shap_vals = explainer.shap_values(self.X_)
        return np.abs(shap_vals[0]).mean(axis=0)

class SHAPNormalizationImportanceRandomForestClassifier(RandomForestClassifier):
    def fit(self, X, y, sample_weight=None):
        self.X_ = X
        super().fit(X, y, sample_weight=sample_weight)
        return self
    @property
    def feature_importances_(self):
        check_is_fitted(self)
        explainer = shap.TreeExplainer(self)
        shap_vals = explainer.shap_values(self.X_)
        return np.abs(shap_vals[0]).mean(axis=0)/(np.abs(shap_vals[0]).mean(axis=0).sum())

class SHAP1ImportanceRandomForestClassifier(RandomForestClassifier):
    def fit(self, X, y, sample_weight=None):
        self.X_ = X[y==1]
        super().fit(X, y, sample_weight=sample_weight)
        return self
    @property
    def feature_importances_(self):
        check_is_fitted(self)
        explainer = shap.TreeExplainer(self)
        shap_vals = explainer.shap_values(self.X_)
        return np.abs(shap_vals[0]).mean(axis=0)

##################################################################

class SHAPImportanceLGBMClassifier(LGBMClassifier):
    def fit(self, X, y, sample_weight=None):
        self.X_ = X
        super().fit(X, y, sample_weight=sample_weight)
        return self
    @property
    def feature_importances_(self):
        check_is_fitted(self)
        explainer = shap.TreeExplainer(self)
        shap_vals = explainer.shap_values(self.X_)
        return np.abs(shap_vals[0]).mean(axis=0)

class SHAP1ImportanceLGBMClassifier(LGBMClassifier):
    def fit(self, X, y, sample_weight=None):
        self.X_ = X[y==1]
        super().fit(X, y, sample_weight=sample_weight)
        return self
    @property
    def feature_importances_(self):
        check_is_fitted(self)
        explainer = shap.TreeExplainer(self)
        shap_vals = explainer.shap_values(self.X_)
        return np.abs(shap_vals[0]).mean(axis=0)

##################################################################

from sklearn.linear_model import LogisticRegression

class SHAPImportanceLogisticRegression(LogisticRegression):
    def fit(self, X, y, sample_weight=None):
        self.X_ = X
        super().fit(X, y, sample_weight=sample_weight)
        return self
    @property
    def feature_importances_(self):
        check_is_fitted(self)
        explainer = shap.LinearExplainer(self, self.X_, feature_perturbation="interventional")
        shap_vals = explainer.shap_values(self.X_)
        return np.abs(shap_vals).mean(axis=0)

class SHAP1ImportanceLogisticRegression(LogisticRegression):
    def fit(self, X, y, sample_weight=None):
        self.X_ = X[y==1]
        super().fit(X, y, sample_weight=sample_weight)
        return self
    @property
    def feature_importances_(self):
        check_is_fitted(self)
        explainer = shap.LinearExplainer(self, self.X_, feature_perturbation="interventional")
        shap_vals = explainer.shap_values(self.X_)
        return np.abs(shap_vals).mean(axis=0)

##################################################################