import numpy as np
from sklearn.model_selection import train_test_split
import os, sys
from matplotlib.pyplot import imread

if sys.version_info[0] == 2:
    from urllib import urlretrieve
    import cPickle as pickle

else:
    from urllib.request import urlretrieve
    import pickle


TRAIN_DATA_LINK = 'https://www.dropbox.com/s/ythu3w0zbq01ixm/trainingSet.tar.gz.zip?dl=0'
TEST_DATA_LINK = 'https://www.dropbox.com/s/zcvn8doi8s48j0p/trainingSample.zip?dl=0'


def unpickle(file):
    with open(file, 'rb') as fo:
        if sys.version_info[0] == 2:
            dict_ = pickle.load(fo)
        else:
            dict_ = pickle.load(fo, encoding='latin1')

    return dict_



def download_cifar(path,
                   url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                   tarname='cifar-10-python.tar.gz', ):
    import tarfile
    if not os.path.exists(path):
        os.mkdir(path)

    urlretrieve(url, os.path.join(path, tarname))
    tfile = tarfile.open(os.path.join(path, tarname))
    tfile.extractall(path=path)


def load_cifar10(data_path=".", test_size=0.2, random_state=1337):
    """
    Basic function to download CIFAR-10 dataset and parse it. Files are located in the internet.
    :param data_path: where to save data
    :param test_size: ratio of images, which will be considered as test
    :param random_state: random state
    :return: X_train, y_train, X_val, y_val, X_test, y_test - numpy arrays
    """
    test_path = os.path.join(data_path, "cifar-10-batches-py/test_batch")
    train_paths = [os.path.join(data_path, "cifar-10-batches-py/data_batch_%i" % i) for i in range(1, 6)]

    if not os.path.exists(test_path) or not all(list(map(os.path.exists, train_paths))):
        print ("Dataset not found. Downloading...")
        download_cifar(data_path,
                       url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                       tarname='cifar-10-python.tar.gz')

    train_batches = list(map(unpickle, train_paths))
    test_batch = unpickle(test_path)

    X = np.concatenate([batch["data"] for batch in train_batches]).reshape([-1, 3, 32, 32]).astype('float32') / 255
    y = np.concatenate([batch["labels"] for batch in train_batches]).astype('int32')
    X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                      test_size=test_size,
                                                      random_state=random_state)

    X_test = test_batch["data"].reshape([-1, 3, 32, 32]).astype('float32') / 255
    y_test = np.array(test_batch["labels"]).astype('int32')

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_mnist(path='./', test_size=0.3, random_state = 123):
    """
    Basic function to download MNIST dataset and parse it. Files are located on our Dropbox.
    :param path: path to directory, where you want to save MNIST images
    :param test_size: ratio of images, which will be considered as test
    :return: X_train, X_test, y_train, y_test - numpy arrays
    """
    
    np.random.seed(random_state)
    if 'X_train.npy' not in os.listdir(path=path) or 'y_train.npy' not in os.listdir(path=path):
        print ("Train dataset not found. Downloading...")
        os.system("curl -L -o train.zip {}".format(TRAIN_DATA_LINK))
        os.system("unzip train.zip")
        os.system("tar -xf trainingSet.tar.gz")
        images = []
        labels = []
        for class_name in os.listdir('./trainingSet'):
            if 'ipynb' not in class_name and '.DS' not in class_name:
                for image_name in os.listdir('./trainingSet/{}'.format(class_name)):
                    image = imread('./trainingSet/{}/{}'.format(class_name, image_name))
                    images.append(image)
                    labels.append(int(class_name))
        X_train = np.array(images)
        y_train = np.array(labels)

        permutation = np.random.permutation(X_train.shape[0])
        X_train = X_train[permutation]
        y_train = y_train[permutation]

        with open('X_train.npy', 'wb') as f:
            np.save(f, X_train)
        with open('y_train.npy', 'wb') as f:
            np.save(f, y_train)
        os.system("rm -rf trainingSet")
        os.system("rm -rf train.zip")
        os.system("rm -rf trainingSet.tar.gz")
    else:
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')

    if 'X_test.npy' not in os.listdir(path=path) or 'y_test.npy' not in os.listdir(path=path):
        print ("Test dataset not found. Downloading...")
        os.system("curl -L -o test.zip {}".format(TEST_DATA_LINK))
        os.system("unzip test.zip")
        os.system("tar -xf trainingSample.tar.gz")
        images = []
        labels = []
        for class_name in os.listdir('./trainingSample'):
            if 'ipynb' not in class_name and '.DS' not in class_name:
                for image_name in os.listdir('./trainingSample/{}'.format(class_name)):
                    image = imread('./trainingSample/{}/{}'.format(class_name, image_name))
                    images.append(image)
                    labels.append(int(class_name))
        X_test = np.array(images)
        y_test = np.array(labels)
        with open('X_test.npy', 'wb') as f:
            np.save(f, X_test)
        with open('y_test.npy', 'wb') as f:
            np.save(f, y_test)

        os.system("rm -rf trainingSample")
        os.system("rm -rf test.zip")
        os.system("rm -rf trainingSet.tar.gz")

    else:
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')

    return X_train, X_test, y_train, y_test



