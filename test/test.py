import sys

sys.path.append("..")
from learner.mdDF import *
import pickle as pkl
from learner.utils import *


def load_pkl(dataset):
    train_path = "../data/" + dataset + "_train.pkl"
    test_path = "../data/" + dataset + "_test.pkl"
    with open(train_path, 'rb') as f:
        train_data, train_label = pkl.load(f)
        train_label = train_label.astype(np.int)
    with open(test_path, 'rb') as f:
        test_data, test_label = pkl.load(f)
        test_label = test_label.astype(np.int)
    return [train_data, train_label, test_data, test_label]


def load_txt(dataset):
    path = "../dataset/{}/".format(dataset)
    train_data = np.loadtxt(path + 'train.txt')
    train_label = np.loadtxt(path + 'label_train.txt', dtype="int")
    test_data = np.loadtxt(path + 'test.txt')
    test_label = np.loadtxt(path + 'label_test.txt', dtype="int")
    return [train_data, train_label, test_data, test_label]


def train_and_fit(dataset, index):
    max_layer = 20
    num_forests = 4
    num_estimator = 100
    gamma = 0.02
    eta = 0.2
    max_depth = 100
    theta = 4
    min_samples_leaf = 1
    n_fold = 5
    percent = -1

    if dataset in ["adult", "yeast", "letter", "HAR", "mnist"]:
        train_data, train_label, test_data, test_label = load_txt(dataset)
    else:
        train_data, train_label, test_data, test_label = load_pkl(dataset)

    clf = Cascade(num_estimator, num_forests, gamma, max_depth, max_layer, theta, eta, min_samples_leaf, n_fold)
    train_p, train_acc, test_p, test_acc, num_layers \
        = clf.train_and_predict([train_data], train_label, [test_data], test_label, index)
    num_layers = min(num_layers, max_layer - 1)
    info1 = "Train accuracy of gcForest on {} is {:.3f}% \n".format(dataset, train_acc[num_layers])
    info2 = "Test  accuracy of gcForest on {} is {:.3f}% \n".format(dataset, test_acc[num_layers])
    print(info1)
    print(info2)


if __name__ == '__main__':
    # datasets = ["adult", "yeast", "letter", "mnist", "sensit", "satimage", "protein"]
    datasets = ["adult"]
    for index, dataset in enumerate(datasets):
        print("\n\n\n")
        print(dataset)
        train_and_fit(dataset, index)
        print("\n\n\n")
