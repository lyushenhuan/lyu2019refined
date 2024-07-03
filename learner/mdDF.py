# from sklearn.model_selection import KFold

from sklearn.cross_validation import KFold
from sklearn.externals import joblib
from .layer import *


def ToOne(X, num_classes):
    num_samples = X.shape[0]
    num_forests = int(X.shape[1] / num_classes)
    b = X.T.reshape([num_forests, num_classes, num_samples])
    c = b.sum(axis=1).reshape([num_forests, 1, num_samples])
    d = b / c
    return d.reshape([num_forests * num_classes, num_samples]).T


##************** SGD ***************************
mu = 0.05  # {0.01, 0.05, 0.1}
r = 0.98  # {0.7, 0.75, 0.8, 0.85, 0.9, 0.95}


def Loss(const_value):
    loss = np.zeros(len(const_value))
    for i, item in enumerate(const_value):
        if item <= r:
            loss[i] = (item - r) ** 2
        else:
            loss[i] = mu * (item - r) ** 2
    return loss


def Gradient(alpha_t, gamma_t, const_value, weight):
    """
    const value = \sum_{i = 0}^{t - 1} {gamma_i * alpha_i}
    """
    grad = np.zeros(len(gamma_t))
    temp1 = 1 / (r ** 2)
    temp2 = mu / ((1 - r) ** 2)
    for i in range(len(gamma_t)):
        z_i = alpha_t * gamma_t[i] + const_value[i]
        if z_i <= r:
            grad[i] = temp1 * z_i
        else:
            grad[i] = temp2 * z_i
    grad = 2 * np.dot(np.multiply(weight, gamma_t), grad)
    return grad


def Optimize(gamma_t, const_value, weight, max_iters=100, eta=0.05):
    alpha_t = np.random.uniform(0, 0.1)
    for i in range(max_iters):
        grad = Gradient(alpha_t, gamma_t, const_value, weight)
        alpha_t = alpha_t - eta * grad
    return alpha_t


##**********************************************
class Cascade:
    def __init__(self, num_estimator, num_forests, gamma, max_depth, max_layer=1, theta=1.1, eta=0.2,
                 min_samples_leaf=1,
                 n_fold=5):
        """
        num_estimator : number of trees in each forest
        num_forests: number of forest in each layer
        gamma: the gamma
        max_layer: stop condition
        max_depth : -1 means we don't consider max_depth
        min_samples_leaf: -1 means we don't consider min_samples_leaf
        n_fold: n-fold cross-validation 
        """
        self.num_estimator = num_estimator
        self.num_forests = num_forests
        self.gamma = gamma
        self.max_depth = max_depth
        self.max_layer = max_layer
        self.theta = theta
        self.min_samples_leaf = min_samples_leaf
        self.n_fold = n_fold
        self.eta = eta

    def train_and_predict(self, train_data_list, train_label, test_data_list, test_label, dataset_index=0):

        train_data_raw = train_data_list[0]
        test_data_raw = test_data_list[0]
        mod = len(train_data_list)

        # basis information of dataset
        num_classes = int(np.max(train_label) + 1)
        num_samples, num_features = train_data_raw.shape

        # basis process
        train_data = train_data_raw.copy()
        test_data = test_data_raw.copy()

        # return value
        train_p = []
        train_acc = []
        val_p = []
        val_acc = []
        test_p = []
        test_acc = []

        # line 1 in algorithm

        best_val_acc = 0.0
        best_t = self.max_layer
        t = 0
        bad = 0

        # line2 in algorithm
        weight = np.ones(num_samples) / num_samples
        const_value = np.zeros(num_samples)
        kf = KFold(len(train_label), n_folds=self.n_fold, shuffle=True)

        # others setting
        alpha_t = 0.0
        g_t = []
        gamma = []
        f_train = []
        f_test = []
        f_val = []
        f_traint = 0
        f_testt = 0
        f_valt = 0

        # while gamma_t > self.gamma:
        while t < self.max_layer:

            print("layer " + str(t))

            self.max_depth = min(self.theta * t + self.theta, 100)

            layer = KfoldWarpper(self.num_forests, self.num_estimator, self.n_fold, kf, t, self.max_depth,
                                 self.min_samples_leaf)

            train_prob, train_stack, val_prob, val_stack, test_prob, test_stack = \
                layer.train_and_predict(train_data, weight, train_label, test_data)

            train_p.append(train_stack)
            val_p.append(val_stack)
            test_p.append(test_stack)

            gamma_t = compute_gamma(val_prob, train_label)
            alpha_t = 0.5 * np.log((1 + gamma_t.mean()) / (1 - gamma_t.mean()))
            # alpha_t = (r * np.sum(gamma_t) - np.dot(gamma_t, const_value)) / np.dot(gamma_t, gamma_t)
            print(alpha_t)
            const_value += alpha_t * gamma_t

            # md-reweighting
            temp_loss = Loss(const_value)
            weight = temp_loss / np.sum(temp_loss)

            g_t = train_stack
            if t > 0:
                f_traint = alpha_t * g_t + f_traint
            else:
                f_traint = alpha_t * g_t
            f_train.append(ToOne(f_traint,num_classes))

            g_t = test_stack
            if t > 0:
                f_testt = alpha_t * g_t + f_testt
            else:
                f_testt = alpha_t * g_t
            f_test.append(ToOne(f_testt,num_classes))

            g_t = val_stack
            if t > 0:
                f_valt = alpha_t * g_t + f_valt
            else:
                f_valt = alpha_t * g_t
            f_val.append(ToOne(f_valt,num_classes))

            pred = np.mean(f_val[t].T.reshape([self.num_forests, num_classes, num_samples]), axis=0).T
            temp_val_acc = compute_accuracy(train_label, pred)
            print("val   acc:" + str(temp_val_acc))

            pred = np.mean(f_train[t].T.reshape([self.num_forests, num_classes, num_samples]), axis=0).T
            temp_train_acc = compute_accuracy(train_label, pred)
            print("train acc:" + str(temp_train_acc))

            pred = np.mean(f_test[t].T.reshape([self.num_forests, num_classes, test_data.shape[0]]), axis=0).T
            temp_test_acc = compute_accuracy(test_label, pred)
            print("test  acc:" + str(temp_test_acc))

            val_acc.append(temp_val_acc)
            train_acc.append(temp_train_acc)
            test_acc.append(temp_test_acc)



            train_data = np.concatenate([train_data_list[(t + 1) % mod], f_val[t]], axis=1)  # LMDF
            test_data = np.concatenate([test_data_list[(t + 1) % mod], f_test[t]], axis=1)

            # train_data = train_data_raw                                       #LMDF_nonpreconc
            # test_data = test_data_raw

            # train_data = f_val[t]                                             #LMDF_stacking
            # test_data = f_test[t]

            t = t + 1

        return [f_train, train_acc, f_test, test_acc, best_t]
