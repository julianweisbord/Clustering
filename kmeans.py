import random
import numpy as np
from sys import maxint
from matplotlib import pyplot as plt

MAX_ITERATIONS = 8

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
    Sum of distance between each row and the centroid of its labeled cluster squared.
    '''
    se = []

    for pos, digit_row in enumerate(data):
        k_centroid_err = []
        for _ in range(k):

            centroid_pos = labels[pos]
            # difference = digit_row - centroids[centroid_pos]
            difference = [np.subtract(x1, x2) for (x1, x2) in zip(digit_row, centroids[centroid_pos])]
            k_centroid_err.append(np.linalg.norm(difference) **2)
        se.append(min(k_centroid_err))
    sse = np.sum(se)
    return sse

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
