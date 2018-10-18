import numpy as np
from sample.starting.person import Person
from sample.starting.person import get_origin_feature
from sample.starting.person import get_look_dict


class LogisticRegression:
    train_count = 300

    def __init__(self):
        self.__W = None

    def __liner__(self, X):
        L = X * self.__W
        return L

    def __exp__(self, X):
        L = -self.__liner__(X)
        exp = np.exp(L)
        return exp

    def __exp_1__(self, X):
        return np.exp(self.__liner__(X))

    def __loss__(self, X, Y):
        exp = self.__exp__(X)
        plus = 1 + exp
        log = np.log(plus)
        F = Y.T * log
        B = (1 - Y.T) * np.log(1 + self.__exp_1__(X))
        return F + B

    def __params__(self, features, labels=None):
        X = np.matrix(features)
        Y = np.matrix(labels) if labels else None
        shape = (X.shape[1] + 1, 1)
        if self.__W is None:
            self.__W = np.matrix([[0] for i in range(shape[0])])
        X = np.insert(X, 0, values=1, axis=1)
        return X, Y

    def __optimize__(self, X, Y, step=.1):
        plus = 1 + self.__exp__(X).I
        res = Y.T - plus
        W = res / X.shape[0] * X
        self.__W = self.__W + step * W.T
        return self

    def train(self, features, labels):
        X, Y = self.__params__(features, labels)
        for i in range(self.train_count):
            self.__optimize__(X, Y)
            loss = self.__loss__(X, Y)
            print(loss)
            print(self.__W)
        return self

    def predict(self, features):
        X, Y = self.__params__(features)
        l1 = 1 + self.__exp__(X)
        return l1.I


if __name__ == '__main__':
    _features = np.random.randint(low=0, high=2, size=(200, 4))
    _labels = [[Person("random", feature).classier().result()] for feature in _features]
    model = LogisticRegression().train(_features, _labels)

    print('======')
    origin = get_origin_feature()
    result = model.predict([v for (k, v) in origin.items()])
    print(result)
