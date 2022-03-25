import warnings 

import numpy as np
import lightgbm as lgb

from scipy.special import expit

# Reference: https://maxhalford.github.io/blog/lightgbm-focal-loss/

class FocalLoss:

    def __init__(self, gamma, alpha=None):
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y):
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    def pt(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true, y_pred):
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    def init_score(self, y_true):
        from scipy import optimize

        res = optimize.minimize_scalar(
            lambda p: self(y_true, p).sum(),
            bounds=(0, 1),
            method='bounded'
        )
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds

    def lgb_obj(self, preds, train_data):
        y = train_data.get_label()
        p = expit(preds)
        return self.grad(y, p), self.hess(y, p)

    def lgb_eval(self, preds, train_data):
        y = train_data.get_label()
        p = expit(preds)
        is_higher_better = False
        return 'focal_loss', self(y, p).mean(), is_higher_better

    
class FocalLossLGBM(lgb.LGBMClassifier):

    def __init__(self, alpha=0.5, gamma=0.1, **kwargs):
        self.alpha = alpha
        self.gamma = gamma
        self.params = kwargs
        self.fl = FocalLoss(alpha=alpha, gamma=gamma)
        
        self._other_params = []
    
    def _fit_optimal_rounds(self, fit_data, max_boost_round, early_stopping_rounds):
        "use this with early_stopping to find optimal number of rounds"

        classifier = lgb.Booster(
            params=self.params, 
            train_set=fit_data,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            results = lgb.cv(
                init_model=classifier,
                params=self.params, 
                train_set=fit_data,
                nfold=2,
                num_boost_round=max_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
                fobj=self.fl.lgb_obj,
                feval=self.fl.lgb_eval
            )
        
#        self.cv_results = results
        return len(results['focal_loss-mean'])

    def fit(self, X_fit, y_fit, max_boost_round=1000, early_stopping_rounds=20):
        
        self.init_score = self.fl.init_score(y_fit)

        fit_data = lgb.Dataset(
            X_fit, y_fit,
            init_score=np.full_like(y_fit, self.init_score, dtype=float),
            free_raw_data=False
        )
        
        self.optimal_boosting_rounds = self._fit_optimal_rounds(fit_data,
                                                                max_boost_round, early_stopping_rounds)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model = lgb.train(
                params=self.params,
                num_boost_round=self.optimal_boosting_rounds,
                train_set=fit_data,
                fobj=self.fl.lgb_obj,
                feval=self.fl.lgb_eval
            )
        
        self.model = model
        return self
        
    def predict_proba(self, X):
        prob_1 =  expit(self.init_score + self.model.predict(X))
        prob_0 = 1 - prob_1
        return np.array([prob_0, prob_1]).T

    def predict(self, X):
        pass

# Testing and benchmarking

if __name__ == '__main__':

    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from lightgbm import LGBMClassifier

    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    def make_balanced_problem(n_samples=5000):
        X, y = make_classification(n_samples=n_samples, n_features=10, n_informative=8, n_redundant=1, n_repeated=1, 
                                   random_state=10) 
        return X, y

    X, y = make_balanced_problem(n_samples=15000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

    for model in [
        FocalLossLGBM(alpha=0.5, gamma=0.5, 
                      learning_rate= 0.1,
                      n_estimators=500,
                      num_leaves=63,
                      verbose=-1, random_state=0),
        HistGradientBoostingClassifier(),
        RandomForestClassifier(),
        LGBMClassifier()
    ]:
        name = type(model).__name__
        print(name)

        model.fit(X_train, y_train)
        y_probs = model.predict_proba(X_test)[:,1]

        print('  AUC  : {0:.3f}'.format(roc_auc_score(y_test, y_probs)))
        print('  AP   : {0:.3f}'.format(average_precision_score(y_test, y_probs)))
        print('  Brier: {0:.3f}'.format(brier_score_loss(y_test, y_probs)))

