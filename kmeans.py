import random
import numpy as np
from sys import maxint
from matplotlib import pyplot as plt

MAX_ITERATIONS = 8
SHOW_IMAGES = False
PLOT = False

def kmeans(data, k):
    # Generate K centroids
    print("K = ", k)
    C, c_positions = gen_centroids(data, k)
    prev_c = 0
    iterations = 0
    SSEs = []
    while not converged(C, prev_c, iterations):
        iterations += 1
        prev_c = C
        labels = label_data(data, C)
        C = gen_new_clusters(data, C, labels, k)
        SSE = calc_sse(data, labels, k, C)
        SSEs.append(SSE)
        print("SSE: ", SSE)
    return SSEs, labels, C, iterations

def converged(C, prev_c, iterations):
    '''
    Check if the difference between the previous centroid and the new centroid has
        changed by more than MARGINAL_DIFFERENCE or if MAX_ITERATIONS is reachead.
    '''
    if not prev_c:
        return False
    if iterations >= MAX_ITERATIONS:
        return True
    # difference = [np.subtract(x1, x2) for (x1, x2) in zip(C, prev_c)]
    # print("difference: ", difference)
    # if difference < .25:
    #     return True

    if np.array_equal(C, prev_c):
        print("No change in iteration!")
        return True


def gen_centroids(data, k):
    centroid_positions = []
    centroids = []
    for _ in range(k):
        c_pos = np.random.randint(0, len(data))
        centroid_positions.append(c_pos)
        centroids.append(data[c_pos])
        print("Centroids: ", centroid_positions)

    return centroids, centroid_positions


def label_data(data, C):
    '''
    For each row in the data, calculate the distance between that
        row and the each centroid.
    input: data <numpy array>, C <list of k numpy arrays>
    return: labels <list> of each centroid number corresponding to
            each row in the data.
    '''
    labels = []
    for n, row in enumerate(data):
        best_dist = maxint
        best_c_k = 0
        for c_k, centroid in enumerate(C):
            row_dist = distance(row, centroid)
            if row_dist < best_dist:
                best_dist = row_dist
                best_c_k = c_k
        labels.append(best_c_k)
    return labels

def gen_new_clusters(data, C, labels, k):

    mean_centroids = []
    sums = []

    for _ in C:
        sums.append([np.zeros(data.shape[1]).flatten(), 0])
    for i, label in enumerate(labels):
        sums[label][0] = np.add(sums[label][0], data[i])
        sums[label][1] += 1
    for i, sample in enumerate(sums):
        mean_centroids.append(np.divide(sample[0], sample[1]))

    return mean_centroids


def distance(point, center):
    return np.linalg.norm(point - center)


def calc_sse(data, labels, k, centroids):
    '''
    description: Sum of distance between each row and the centroid of its labeled cluster squared.
    input: data <list of lists> the dataset, labels <list of ints> are a series of values
               which corresponds to each row in data, k <int> number of centroids,
               centroids <list of k centroid lists> the current centroid sample.
    return: sse <list> Sum of squared error for each iteration of kmeans.
    '''
    se = []

    for pos, digit_row in enumerate(data):
        k_centroid_err = []
        for _ in range(k):

            centroid_pos = labels[pos]
            difference = [np.subtract(x1, x2) for (x1, x2) in zip(digit_row, centroids[centroid_pos])]
            k_centroid_err.append(np.linalg.norm(difference) **2)
        se.append(min(k_centroid_err))
    sse = np.sum(se)
    return sse

def compute_eigens(data):
    data = np.array(data)
    mean = np.mean(data)
    mean_img = np.mean(data, axis=0)
    datap = (data-np.mean(data.T, axis=1)).T
    cov = np.cov(datap)
    # print("Covariance Matrix: ", cov)
    w, v = np.linalg.eig(cov)
    # print("Eigenvectors: ", v)
    print("Covariance shape", cov.shape)
    print("shape of w:", w.shape)
    print("shape of v:", v.shape)
    print("Covariance shape[1]", cov[1].shape)
    # abs_w = [abs(i) for i in w]
    # print("type(abs_w): ", type(abs_w))
    # print("abs_w: ", abs_w)
    eig_vals = np.sort(w)
    eig_val_indexs = np.argsort(w)
    eig_val_lrgst_indexs = eig_val_indexs[-10:]

    # eig_vals = np.sort(abs_w)
    print("type(eig_vals): ", type(eig_vals))
    # print("eig_vals: ", eig_vals[-10:])
    largest_eig_vals = eig_vals[-10:]
    largest_eig_decreasing = []
    eig_lrgst_indexs_decreasing = []
    for eig in range(1, 11):
        largest_eig_decreasing.append(largest_eig_vals[-eig])
        eig_lrgst_indexs_decreasing.append(eig_val_lrgst_indexs[-eig])
    largest_eig_decreasing = np.array(largest_eig_decreasing)
    print("Largest Eig Values Decreasing: ", largest_eig_decreasing)

    largest_eigvector_decreasing = []
    for i in eig_lrgst_indexs_decreasing:
        largest_eigvector_decreasing.append(v[:,i])
    print("In order eigen vectors[0]: ", largest_eigvector_decreasing[0])
    if SHOW_IMAGES:
        import cv2
        print("Shape of eigenvector: ", largest_eigvector_decreasing[0].shape)
        # largest_eigvector_decreasing = np.divide(largest_eigvector_decreasing, np.absolute(largest_eigvector_decreasing[-1]))
        for pos, ev in enumerate(largest_eigvector_decreasing):
            ev = np.reshape(ev, (28, 28))
            ev_rescale = cv2.normalize(np.absolute(ev), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.namedWindow("Eigenvector", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Eigenvector', 600,600)
            cv2.imshow("Eigenvector", ev_rescale)
            cv2.waitKey(0)

        mean_img = np.reshape(mean_img, (28, 28))
        mean_img_rescale = cv2.normalize(np.absolute(mean_img), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.namedWindow("Mean Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Mean Image', 600, 600)
        cv2.imshow("Mean Image", mean_img_rescale)
        cv2.waitKey(0)

    return largest_eig_decreasing

def project(lrg_eigenvectors, data):

    if SHOW_IMAGES:
        import cv2
        prev_i = 0
        for i in range(10, len(data), 10):
            projected = lrg_eigenvectors.T.dot(data[prev_i:i])

            print("projected.shape: ", projected.shape)
            project_img = np.reshape(projected, (28, 28))
            project_img_rescale = cv2.normalize(np.absolute(project_img), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.namedWindow("Projected Image", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Projected Image', 600, 600)
            cv2.imshow("Projected Image", project_img_rescale)
            cv2.waitKey(0)
            prev_i = i


def plot_sse(sses, iterations):
    if PLOT:
        if isinstance(iterations, int):
            iterations = [i for i in range(1, iterations + 1)]
            iter_str = "Iterations"
        else:
            iterations = [i for i in range(iterations[0], iterations[1])]
            iter_str = "K values"

        print("sses: ", sses)
        print("iterations", iterations)
        plt.plot(iterations, sses)
        plt.xlabel(iter_str)
        plt.ylabel("SSE's")
        plt.show()
