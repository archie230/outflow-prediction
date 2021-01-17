import numpy as np
from sklearn.base import BaseEstimator


def entropy(y, EPS=0.0005):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """

    probas = np.mean(y, axis=0)
    log_probas = np.log(probas + EPS)
    
    return -np.sum(probas * log_probas)
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    p = np.mean(y, axis=0)
    return 1 - np.sum(p ** 2)
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    return np.var(y)

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    return np.sum(abs(y - np.median(y))) / len(y)


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    Class of tree node
    """
    def __init__(self, feature_index, threshold, depth=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = None
        self.left_child = None
        self.right_child = None
        self.depth = depth
        self.isleaf = False
        
class DecisionTree(BaseEstimator):
    # (criterion, classification flag)
    all_criterions = {
        'gini': (gini, True),
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None 
        self.debug = debug
        self.ans = 0
        self.prob_ans = None

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        mask = X_subset[:, feature_index] < threshold
        opposite_mask = X_subset[:, feature_index] >= threshold

        y_left = y_subset[mask]
        y_right = y_subset[opposite_mask]
        
        X_left = X_subset[mask]
        X_right = X_subset[opposite_mask]
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        mask = X_subset[:, feature_index] < threshold
        opposite_mask = X_subset[:, feature_index] >= threshold
        y_left = y_subset[mask]
        y_right = y_subset[opposite_mask]
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        best_criterion_value = np.inf
        best_feature, best_threshold = 0, 0

        for feature_id in range(X_subset.shape[1]):
            features = self.feature_values[feature_id]

            for threshold in features:
                y_left, y_right = self.make_split_only_y(feature_id, threshold, X_subset, y_subset)

                if y_left.shape[0] == 0 or y_right.shape[0] == 0:
                    continue

                criterion_value = self.criterion(y_left) * y_left.shape[0] + \
                                  self.criterion(y_right) * y_right.shape[0]

                if criterion_value < best_criterion_value:
                    best_feature = feature_id
                    best_threshold = threshold
                    best_criterion_value = criterion_value

        return best_feature, best_threshold
    
    def make_tree(self, X_subset, y_subset, depth=0):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        if depth > self.depth:
            self.depth = depth
        
        if X_subset.shape[0] <= self.min_samples_split or depth == self.max_depth or len(set(y_subset.flatten())) == 1:
            new_node = Node(0, 0, depth=depth)
            new_node.isleaf = True
            if not self.all_criterions[self.criterion_name][1]:
                new_node.value = np.mean(y_subset)
            new_node.proba = np.mean(y_subset, axis=0)
            return new_node
        
            
        opt_index, opt_threshold = self.choose_best_split(X_subset, y_subset)
        (X_left, y_left), (X_right, y_right) = self.make_split(opt_index, opt_threshold, X_subset, y_subset)
        
        new_node = Node(opt_index, opt_threshold, depth)
        if X_left.shape[0] == 0 or X_right.shape[0] == 0: 
            new_node.isleaf = True
           
            if not self.all_criterions[self.criterion_name][1]:
                new_node.value = np.mean(y_subset)
            
                
        new_node.left_child = self.make_tree(X_left, y_left, depth+1)
        new_node.right_child = self.make_tree(X_right, y_right, depth+1)    
                
        new_node.proba = np.mean(y_subset, axis=0)
        return new_node
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        
        self.feature_values = []
        for feature_id in range(X.shape[1]):
            thresholds = np.sort(np.unique(X[:, feature_id]))
            if len(thresholds) < 50:
                self.feature_values.append(thresholds)
            else:
                thresholds = [thresholds[int(i * len(thresholds) / 50.)] for i in range(50)]
                self.feature_values.append(thresholds)
        
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y, 1)
    
    def walk(self, x, node):
        if node.isleaf:
            if not self.all_criterions[self.criterion_name][1]:
                self.ans = node.value
            else: 
                self.ans = np.argmax(node.proba)
        else:
            if x[node.feature_index] < node.value:
                self.walk(x, node.left_child)
            else:
                self.walk(x, node.right_child)
    
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

        y_predicted = []
        
        for i in range(X.shape[0]):
            self.walk(X[i], self.root)
            y_predicted.append(self.ans)
            
        y_predicted = np.array(y_predicted)
        return y_predicted
    
    def walk_proba(self, x, node):
        if node.isleaf:
            self.prob_ans = node.proba
            
        else:
            if x[node.feature_index] < node.value:
                self.walk_proba(x, node.left_child)
            else:
                self.walk_proba(x, node.right_child)
        
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

        y_predicted_probs = []
        
        for i in range(X.shape[0]):
            self.walk_proba(X[i], self.root)
            y_predicted_probs.append(self.prob_ans)              
            
        y_predicted_probs = np.array(y_predicted_probs)
        return y_predicted_probs
