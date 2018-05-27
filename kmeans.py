import random
import numpy as np
from sys import maxint
from matplotlib import pyplot as plt

MAX_ITERATIONS = 8
SHOW_IMAGES = True

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
        # print("len centroids: ", len(centroids))
        # print("centroid_positions", centroids)
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

    sums = []
    mean_centroids = []

    for centroid in C:
        sums.append([np.zeros(data.shape[1]).flatten(), 0])

    for i, label in enumerate(labels):
        sums[label][0] = np.add(sums[label][0], data[i])
        sums[label][1] += 1

    for i, row in enumerate(sums):
        mean_centroids.append(np.divide(row[0], row[1]))

    return mean_centroids
    # new_clusters = []
    # # Identify each samples closest cluster
    # for cluster in range(k):
    #     points_around_cluster = []
    #     for sample_pos, sample in enumerate(data):
    #         if labels[sample_pos] != cluster:
    #             continue
    #         points_around_cluster.append(sample)
    #     # Sum elementwise all assigned samples of a cluster and divide by total num samples
    #     # Assign cluster to that vector.
    #     summed_samples = [sum(i) for i in zip(*points_around_cluster)]
    #     new_clusters.append(summed_samples)

    # return new_clusters


def distance(point, center):
    return np.linalg.norm(point - center)

def get_k():
    pass

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

def compute_covariance(data):
    data = np.array(data)
    mean = np.mean(data)
    # n = len(data)
    # covariance_matrix = []
    # for sample in data:
    #     difference = sample - mean
    #     covariance_matrix.append(np.dot(difference, difference.T))
    #     m = np.dot(difference, difference.T)
    #     print(difference)
    #     # print("covariance_matrix: ", covariance_matrix)
    # # sum_cov = np.sum(covariance_matrix)
    # sum_cov = covariance_matrix
    # # sum_cov = np.divide(np.array(sum_cov), n)
    # # print("shape: ", sum_cov.shape)
    # print("shape cov: ", np.cov(data))
    # return sum_cov

    # X = data
    # avg = average(X, axis=1, returned=True)
    # X-= avg[:, None]
    # X_T = X.T
    # fact = X.shape[1] - 1
    # c = np.dot(X, X_T.conj())
    # c *= 1. / np.float64(fact)
    # return c.squeeze()
    cov = np.cov(data)
    # print("Covariance Matrix: ", cov)
    w, v = np.linalg.eig(cov)
    # print("Eigenvectors: ", v)
    print("Covariance shape", cov.shape)
    print("shape of w:", w.shape)
    print("shape of v:", v.shape)
    print("Covariance shape[0]", cov[0].shape)
    # abs_w = [abs(i) for i in w]
    # print("type(abs_w): ", type(abs_w))
    # print("abs_w: ", abs_w)
    eig_vals = np.sort(w)
    eig_val_indexs = np.argsort(w)
    eig_val_lrgst_indexs = eig_val_indexs[-10:]

    # eig_vals = np.sort(abs_w)
    print("type(eig_vals): ", type(eig_vals))
    print("eig_vals: ", eig_vals[-10:])
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
    print("In order eigen vectors: ", largest_eigvector_decreasing)
    if SHOW_IMAGES:
        import cv2
        print("Shape of eigenvector: ", largest_eigvector_decreasing[0].shape)
        print("Shape of eigenvector[0]: ", largest_eigvector_decreasing[0][0].shape)
        for pos, ev in enumerate(largest_eigvector_decreasing):
            ev[0] = np.reshape(ev[0], (28, 28))
            ev[0] = cv2.normalize(ev[0], norm_type=cv2.NORM_MINMAX)
            cv2.imshow("Eigenvector{}".format(pos), ev[0])
            cv2.waitKey(0)
    # plt.imshow(np.reshape(v[0][0], (28, 28)))
    # for i in range(len(w)):
    #     print("Value: ", w[i])
    #     print("Corresponding Vector: ", v[:, i])
    return largest_eig_decreasing



def plot_sse(sses, iterations):
    if isinstance(iterations, int):
        iterations = [i for i in range(1, iterations + 1)]
    else:
        iterations = [i for i in range(iterations[0], iterations[1])]

    print("sses: ", sses)
    print("iterations", iterations)
    plt.plot(iterations, sses)
    plt.xlabel("Iterations")
    plt.ylabel("SSE's")
    plt.show()
