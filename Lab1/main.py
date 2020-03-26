from keras.datasets import mnist
import numpy as np

from PIL import Image


def invert_pixels(img):
    result = np.copy(img)
    (n, m) = img.shape
    for i in range(m):
        for j in range(m):
            if result[i][j] == 0:
                result[i][j] = 1
            else:
                result[i][j] = -1
    return result


def invert_mnist_pixels(img):
    result = np.copy(img)
    (n, m) = img.shape
    for i in range(m):
        for j in range(m):
            if result[i][j] == 0:
                result[i][j] = -1
    return result

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


def init_network(etalons, e):
    (n, m, l) = etalons.shape

    # init ideal numbers shapes

    # init weights
    X = []

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
    # print(W)
    # print(E)
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
    y_next = map_activation(np.copy(s1), M / 2)

    Emax = 0.1
    while True:
        y_prev = np.copy(y_next)
        y_next = map_activation(e.dot(y_prev), M / 2)
        if not np.linalg.norm(y_prev - y_next) > Emax:
            break
    return y_next


def main():
    print(y_train[4])
    img = x_train[4].flatten()

    # init ideal numbers shapes
    etalons = []

    for num in range(10):
        img = Image.open("images/" + str(num) + ".bmp").convert("L")
        etalons.append(np.array(img))
    etalons = np.around(np.array(etalons).astype(np.float32) / 255).astype(np.int)
    (n, m, l) = etalons.shape
    for k in range(n):
        etalons[k] = invert_pixels(etalons[k])
    # test_shape = invert_pixels((np.array(Image.open("images/0.bmp").convert("L")).astype(np.float32) / 255)
    #                            .astype(np.int)).flatten()
    # network_test(*init_network(etalons, 0.01), test_shape)
    #
    # test_shape = invert_pixels((np.array(Image.open("images/1.bmp").convert("L")).astype(np.float32) / 255)
    #                            .astype(np.int)).flatten()
    # network_test(*init_network(etalons, 0.01), test_shape)
    #
    # test_shape = invert_pixels((np.array(Image.open("images/2.bmp").convert("L")).astype(np.float32) / 255)
    #                            .astype(np.int)).flatten()
    # network_test(*init_network(etalons, 0.01), test_shape)
    #
    # test_shape = invert_pixels((np.array(Image.open("images/3.bmp").convert("L")).astype(np.float32) / 255)
    #                            .astype(np.int)).flatten()
    # network_test(*init_network(etalons, 0.01), test_shape)
    #
    # test_shape = invert_pixels((np.array(Image.open("images/4.bmp").convert("L")).astype(np.float32) / 255)
    #                            .astype(np.int)).flatten()
    # network_test(*init_network(etalons, 0.01), test_shape)
    #
    # test_shape = invert_pixels((np.array(Image.open("images/5.bmp").convert("L")).astype(np.float32) / 255)
    #                            .astype(np.int)).flatten()
    # network_test(*init_network(etalons, 0.01), test_shape)

    test_shape = invert_pixels((np.array(Image.open("test_images/2.bmp").convert("L")).astype(np.float32) / 255)
                               .astype(np.int)).flatten()
    print(network_test(*init_network(etalons, 0.01), test_shape))

    Image.open("test_images/2.bmp").convert("L").convert("1").save("wb_2.bmp")

    (img_count, n, m) = x_train.shape

    (w, e) = init_network(etalons, 0.01)
    success = 0
    successes = []
    for k in range(img_count):
        inv_img = invert_mnist_pixels(x_train[k]).flatten()
        if np.argmax(network_test(w, e, inv_img)) == y_train[k]:
            success += 1
            successes.append(k)
            print(str(success) + " ," + str(k))
    print(success)
    print(success / img_count)
    print(successes)
    successes = np.array(successes)
    suc_n = successes.shape[0]
    for i in range(suc_n):
        print(network_test(w, e, invert_mnist_pixels(x_train[successes[i]]).flatten()))

    (x_train_test, y_train_test), (x_test_test, y_test_test) = mnist.load_data()
    Image.fromarray(x_train_test[4]).convert("L").save("test.bmp")

    etalons = [
        [[1., 1., 1.],  [1., -1., -1.],  [1., -1., -1.],  [1., -1., -1.],  [1., 1., 1.]],
        [[1., 1., 1.],  [1., -1., -1.],  [1., 1., -1.],   [1., -1., -1.],  [1., 1., 1.]],
        [[1., 1., 1.],  [1., 1., 1.],   [1., -1., 1.],   [1., -1., 1.],   [1., 1., 1.]]
    ]
    etalons = np.array(etalons)

    # 1
    test_shape = [1., 1., 1.,  1., -1., -1.,  1., -1., -1.,  1., -1., -1.,  1., 1., 1.]

    # # 2
    # test_shape = [1., 1., 1.,  1., -1., -1.,  1., 1., -1.,   1., -1., -1.,  1., 1., 1.]

    # # 3
    # test_shape = [1., 1., 1.,  1., 1., 1.,   1., -1., 1.,   1., -1., 1.,   1., 1., 1.]


    # test_shape = [1., 1., 1.,  1., -1., 1.,   1., -1., 1.,   1., -1., 1.,   1., 1., 1.]
    test_shape = np.array(test_shape)

    network_test(*init_network(etalons, 0.3), test_shape)

main()
