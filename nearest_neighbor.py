import pickle
import numpy as np
import os
import time
import dask.array as da
from collections import Counter


def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding="latin1")
    return data


def classification_accuracy(prediction, ground_truth):
    ground_truth = ground_truth[:prediction.shape[0]]
    n_images = prediction.shape[0]
    x = prediction - ground_truth
    n_wrong_predictions = np.count_nonzero(x)
    accuracy = (n_images - n_wrong_predictions) / n_images

    return accuracy*100


def load_training_data(dataset_path):

    train_images = np.zeros([50000, 3072])
    train_labels = np.zeros([50000])

    start = 0
    n_images_in_a_file = 10000
    for i in range(1, 6):
        path = os.path.join(dataset_path, "data_batch_{}".format(i))
        data_dict = unpickle(path)
        train_images[start: start + n_images_in_a_file, :] = data_dict["data"]
        train_labels[start: start + n_images_in_a_file] = data_dict["labels"]
        start += n_images_in_a_file

    return np.asarray(train_images, dtype=np.int), np.asarray(train_labels, dtype=np.int)


def load_test_data(dataset_path):
    path = os.path.join(dataset_path, "test_batch")
    datadict = unpickle(path)
    test_images = datadict["data"]
    test_labels = datadict["labels"]
    return np.asarray(test_images, dtype=np.int), np.asarray(test_labels, dtype=np.int)


def random_classifier(n_images=10000):
    np.random.seed(42)
    random_prediction = np.random.randint(0, 10, n_images)
    return random_prediction


def nearest_neighbour(test_images, train_images, train_labels, k=1):
    pred = np.zeros(test_images.shape[0])
    for i in range(test_images.shape[0]):
        test_image = test_images[i, :]
        nn = da.sum(np.abs(train_images - test_image), axis=1, keepdims=True)
        if k == 1:
            nn = da.argmin(nn, axis=0)
            pred[i] = train_labels[nn]
        else:
            nn = np.array(nn)
            min_idx = np.argsort(nn, 0)[:k]
            labels = np.array([train_labels[i] for i in min_idx])
            labels = np.reshape(labels, [-1])
            lab = Counter(labels).most_common()[0][0]
            pred[i] = lab
    return pred


def main():

    datset_path = "cifar-10-batches-py"
    # time for loading
    st = time.time()
    train_images, train_labels = load_training_data(datset_path)
    test_images, test_labels = load_test_data(datset_path)
    et = time.time()
    print("Time taken for loading images = {}".format(et-st))

    random_prediction = random_classifier(10000)
    random_accuracy = classification_accuracy(random_prediction, test_labels)
    print("Random classifier accuracy = {}".format(random_accuracy))

    #  ################ Naive implementation for 1NN classifier ########################
    st = time.time()
    k = 1
    test_images, train_images, train_labels = da.array(test_images), da.array(train_images), da.array(train_labels)
    prediction = nearest_neighbour(test_images, train_images, train_labels, k=k)
    accuracy = classification_accuracy(prediction, test_labels)
    et = time.time()
    print("{} nearest neighbor classifier accuracy = {}".format(k, accuracy))
    print("Time taken for classifying test images = {}".format(et - st))


if __name__ == '__main__':
    main()
