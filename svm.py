from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import KFold
import numpy as np
import warnings

# "poly": Polynominal, "rbf": RBF
class SupportVectorClassifier:
    def __init__(self, loss: str = "hinge",
                 penalty: str = "l2",
                 kernel: str = "linear"):

        """
        Will use LinearSVC if kernel = "linear"
        """
        self.loss = loss
        self.penalty = penalty
        self.kernel = kernel
        

    def create_model(self, C):
        model = None
        if self.kernel == "linear":
            model = LinearSVC(penalty=self.penalty, loss = self.loss, C=C)
        else:
            model = SVC(kernel=self.kernel, C=C)

        return model


    def kfold_cross_validation(self, X_train, Y_train, k, Cs):
        kfold = KFold(n_splits=k)
        validation_err = []
        training_err = []

        for C in Cs:
            val_errors = []
            training_errors = []

            for train_idx, val_idx in kfold.split(X_train):
        
                classifier = self.create_model(C)

                X_train_split, Y_train_split = X_train[train_idx, :], Y_train[train_idx]
                X_val_split, Y_val_split = X_train[val_idx, :], Y_train[val_idx]  

                classifier.fit(X_train_split, Y_train_split)

                training_errors.append(1 - classifier.score(X_train_split, Y_train_split))
                val_errors.append(1 - classifier.score(X_val_split, Y_val_split))
            
            validation_err.append(np.mean(val_errors))
            training_err.append(np.mean(training_errors))

        best_C = Cs[np.argmin(validation_err)]

        return training_err, validation_err, best_C
    

    def train(self, X_train, Y_train):
        self.classifier = self.create_model(C)
        self.classifier.fit(X_train, Y_train)
    

    def eval(self, X_test, Y_test):
        if self.classifier:
            error_rate = self.classifier.score(X_test, Y_test)
            return error_rate
        return None
