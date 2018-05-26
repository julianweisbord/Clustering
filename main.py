import sys
import numpy as np
import kmeans as km

DATA_PATH = "./data-1.txt"

def main():
    if len(sys.argv) == 2:
        fl = sys.argv[1]
    else:
        fl = DATA_PATH
    data = load_data(fl)
    # print("data[0]", data[0])

    # k = 2
    # SSEs, labels, C, iterations = km.kmeans(data, k)
    # km.plot_sse(SSEs, iterations)

    lowest_sses = []
    for k in range(2, 11):
        sses_k = []
        for i in range(10):
            SSEs, labels, C, iterations = km.kmeans(data, k)
            sses_k.append(SSEs)
        lowest_sses.append(min(sses_k))
        print("Lowest SSE for k = {}: {}".format(k, lowest_sses[-1]))

    km.plot_sse(lowest_sses, [2,11])


def load_data(fl):
    return np.genfromtxt(fl, delimiter=',')

if __name__ == '__main__':
    main()
