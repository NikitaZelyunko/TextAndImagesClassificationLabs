from keras.datasets import mnist
import numpy as np

from PIL import Image


def get_activation_func(t_param):
    def activation(value):
        if value < 0:
            return 0
        if value > t_param:
            return t_param
        return value

    return activation


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.around(x_train.astype(np.float32) / 255).astype(np.int)


def init_network(e):
    # init ideal numbers shapes
    etalons = []

    for num in range(10):
        img = Image.open("images/" + str(num) + ".bmp").convert("L")
        etalons.append(np.array(img))
    etalons = np.around(np.array(etalons).astype(np.float32) / 255).astype(np.int)
    # init ideal numbers shapes

    # init weights
    X = []
    (n, m, l) = etalons.shape
    w = []
    for num in range(n):
        X.append(etalons[num].flatten())

    M = m * l
    W = np.zeros((n, M))

    for i in range(n):
        for j in range(M):
            W[i][j] = X[i][j] / 2

    # init relations matrix
    E = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                E[i][j] = -e
            else:
                E[i][j] = 1
    # init relations matrix
    return W, E


def map_activation(arr, t_param):
    n = arr.shape[0]
    result = np.zeros(n)
    activation = get_activation_func(t_param)
    for i in range(n):
        result[i] = activation(arr[i])
    return result


def network_test(w, e, value):
    n, M = w.shape
    s1 = w.dot(value)
    print("test", s1.shape)
    s_next = map_activation(np.copy(s1), M / 2)
    s_prev = np.copy(s_next)

    Emax = 0.1
    while True:
        
        if not np.linalg.norm(s_prev - s_next) > Emax:
            break


def main():
    img = x_train[0].flatten()
    network_test(*init_network(0.01), img)


main()
