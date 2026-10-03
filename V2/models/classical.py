from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class ClassicalMLSuite:
    @staticmethod
    def get_logistic_regression(random_state: int = 42):
        return LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=random_state)

    @staticmethod
    def get_decision_tree(max_depth: int = 6, random_state: int = 42):
        return DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", criterion="gini", random_state=random_state)

    @staticmethod
    def get_random_forest(n_estimators: int = 60, max_depth: int = 8, random_state: int = 42):
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight="balanced", n_jobs=-1, random_state=random_state)

    @staticmethod
    def get_svm(C: float = 1.0, gamma: str = "scale", random_state: int = 42):
        return SVC(C=C, kernel="rbf", gamma=gamma, probability=True, class_weight="balanced", random_state=random_state)
