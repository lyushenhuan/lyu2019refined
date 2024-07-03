from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

from .utils import *


class Layer:
    def __init__(self, n_estimators, num_forests, num_classes, max_depth=100, min_samples_leaf=1):
        self.num_forests = num_forests  # number of forests
        self.n_estimators = n_estimators  # number of trees in each forest
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def train_and_predict(self, train_data, weight, train_label, val_data, test_data):
        predict_prob = np.zeros([self.num_forests, test_data.shape[0], self.num_classes])
        val_prob = np.zeros([self.num_forests, val_data.shape[0], self.num_classes])
        train_prob = np.zeros([self.num_forests, train_data.shape[0], self.num_classes])

        for forest_index in range(self.num_forests):
            predict_p = np.zeros([test_data.shape[0], self.num_classes])
            val_p = np.zeros([val_data.shape[0], self.num_classes])
            train_p = np.zeros([train_data.shape[0], self.num_classes])
            if forest_index % 2 == 0:
                clf = RandomForestClassifier(n_estimators=self.n_estimators,
                                             n_jobs=-1, max_features="sqrt",
                                             max_depth=self.max_depth,
                                             min_samples_leaf=self.min_samples_leaf)
                clf.fit(train_data, train_label, weight)
                # clf.fit(train_data, train_label)
                temp = clf.predict_proba(train_data)
                train_p += temp
                temp = clf.predict_proba(val_data)
                val_p += temp
                temp = clf.predict_proba(test_data)
                predict_p += temp
            else:
                clf = ExtraTreesClassifier(n_estimators=self.n_estimators,
                                           n_jobs=-1, max_features="sqrt",
                                           max_depth=self.max_depth,
                                           min_samples_leaf=self.min_samples_leaf)
                clf.fit(train_data, train_label, weight)
                # clf.fit(train_data, train_label)
                temp = clf.predict_proba(train_data)
                train_p += temp
                temp = clf.predict_proba(val_data)
                val_p += temp
                temp = clf.predict_proba(test_data)
                predict_p += temp

            train_prob[forest_index, :] = train_p
            val_prob[forest_index, :] = val_p
            predict_prob[forest_index, :] = predict_p

        train_avg = np.sum(train_prob, axis=0)
        train_avg /= self.num_forests
        train_concatenate = train_prob.transpose((1, 0, 2))
        train_concatenate = train_concatenate.reshape(train_concatenate.shape[0], -1)

        val_avg = np.sum(val_prob, axis=0)
        val_avg /= self.num_forests
        val_concatenate = val_prob.transpose((1, 0, 2))
        val_concatenate = val_concatenate.reshape(val_concatenate.shape[0], -1)

        predict_avg = np.sum(predict_prob, axis=0)
        predict_avg /= self.num_forests
        predict_concatenate = predict_prob.transpose((1, 0, 2))
        predict_concatenate = predict_concatenate.reshape(predict_concatenate.shape[0], -1)

        return [train_avg, train_concatenate, val_avg, val_concatenate, predict_avg, predict_concatenate]


class KfoldWarpper:
    def __init__(self, num_forests, n_estimators, n_fold, kf, layer_index, max_depth=31, min_samples_leaf=1):
        self.num_forests = num_forests
        self.n_estimators = n_estimators
        self.n_fold = n_fold
        self.kf = kf
        self.layer_index = layer_index
        self.max_depth = max_depth
        self.num_classes = 2
        self.min_samples_leaf = min_samples_leaf

    def train_and_predict(self, train_data, weight, train_label, test_data):
        self.num_classes = int(np.max(train_label) + 1)
        num_samples, num_features = train_data.shape

        train_prob = np.zeros([num_samples, self.num_classes])
        train_prob_concatenate = np.zeros([num_samples, self.num_forests * self.num_classes])

        val_prob = np.empty([num_samples, self.num_classes])
        val_prob_concatenate = np.empty([num_samples, self.num_forests * self.num_classes])

        test_prob = np.zeros([test_data.shape[0], self.num_classes])
        test_prob_concatenate = np.zeros([test_data.shape[0], self.num_forests * self.num_classes])

        fold = 0
        for train_index, test_index in self.kf:
            X_train = train_data[train_index, :]
            X_val = train_data[test_index, :]
            y_train = train_label[train_index]

            # training fold-th layer
            layer = Layer(self.n_estimators, self.num_forests, self.num_classes, self.max_depth, self.min_samples_leaf)
            # predict_prob val
            temp1, temp2, val_prob[test_index], \
            val_prob_concatenate[test_index, :], \
            temp_prob, temp_prob_concatenate = \
                layer.train_and_predict(X_train, weight[train_index],
                                        y_train, X_val, test_data)

            train_prob[train_index] += temp1
            train_prob_concatenate[train_index, :] += temp2
            test_prob += temp_prob
            test_prob_concatenate += temp_prob_concatenate
            fold += 1

        test_prob /= self.n_fold
        test_prob_concatenate /= self.n_fold
        train_prob /= (self.n_fold - 1)
        train_prob_concatenate /= (self.n_fold - 1)

        return [train_prob, train_prob_concatenate, val_prob, val_prob_concatenate, test_prob, test_prob_concatenate]
