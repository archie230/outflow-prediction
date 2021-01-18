import numpy as np
from sklearn.base import BaseEstimator
from tree import DecisionTree

class RandomForest(BaseEstimator):
    # classification flag
    all_criterions = {
        'gini': True,
        'entropy': True,
        'variance': False,
        'mad_median': False
    }
    
    def __init__(self, n_estimators=100, criterion_name='gini',
                 max_depth=np.inf, min_samples_split=2,
                 bootstrap=True, random_state=42):
        self.n_estimators = n_estimators
        self.criterion_name = criterion_name
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.random_state = random_state
        
        
        self.estimators = [
            DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                         criterion_name=self.criterion_name, random_state=self.random_state) 
            for _ in range(self.n_estimators)
        ]
        
    def get_bootstrap(self, X, y):
        if self.bootstrap:
            idx = np.random.choice(range(X.shape[0]), X.shape[0])
            return (X[idx, :], y[idx, :])
        else:
            return (X, y)
        
    def fit(self, X, y, random_ft_subspace_sz=None):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
            
        random_ft_subspace_sz : int features subspace size
        """
        
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        assert random_ft_subspace_sz is None or random_ft_subspace_sz <= X.shape[1], \
            'Subspace size should be less or equal to features num'
        
        self.classification = self.all_criterions[self.criterion_name]
        
        if random_ft_subspace_sz is None:
            if self.classification:
                random_ft_subspace_sz = int(np.sqrt(X.shape[1]))
            else:
                random_ft_subspace_sz = int(X.shape[1])
        
        for estimator in self.estimators:
            estimator.fit(*self.get_bootstrap(X, y), 
                          random_ft_subspace_sz=random_ft_subspace_sz)
            
        return self
    
    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        if self.classification:
            proba = self.predict_proba(X)
            y_pred = np.argmax(proba, axis=1)
        else:
            y_pred = np.array([estimator.predict(X) for estimator in self.estimators])
            y_pred = np.mean(y_pred, axis=0)
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'
        
        proba = np.array([estimator.predict_proba(X) for estimator in estimators])
        proba = np.mean(proba, axis=0)
        
        return proba
        
        
        
        
        
        
        