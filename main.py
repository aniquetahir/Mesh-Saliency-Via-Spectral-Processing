import argparse
import trimesh
import numpy as np
from tqdm import tqdm
from util import get_laplacian
# from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pickle
import os
from trimesh.smoothing import laplacian_calculation

parser = argparse.ArgumentParser(description='Calculate the saliency of a mesh')
parser.add_argument('-i', required=True, type=str, help='The input mesh location')

if __name__ == "__main__":
    # args = parser.parse_args()
    # filename = args.i
    filename = 'models/104.off'

    # Load the model
    mesh = trimesh.load(filename)

    # Initialize adjacency matrix
    num_vertices = len(mesh.vertices)
    print('%d vertices found' % num_vertices)
    if os.path.exists('tmp/A.pickle'):
        with open('tmp/A.pickle', 'rb') as A_file:
            A = pickle.load(A_file)
    else:
        A = np.zeros((num_vertices,  num_vertices))

        # Populate adjacency matrix
        edges = list(mesh.vertex_adjacency_graph.edges)
        for edge in edges:
            A[edge[0], edge[1]] = 1
            A[edge[1], edge[0]] = 1
        with open('tmp/A.pickle', 'wb') as A_file:
            pickle.dump(A, A_file)

    # Create Diagonal degree matrix
    if os.path.exists('tmp/D.pickle'):
        with open('tmp/D.pickle', 'rb') as D_file:
            D = pickle.load(D_file)
    else:
        D = np.zeros((num_vertices, num_vertices))
        for i in range(num_vertices):
            D[i, i] = np.sum(A[i, :])
        with open('tmp/D.pickle', 'wb') as D_file:
            pickle.dump(D, D_file)



    # Create the weight matrix
    if os.path.exists('tmp/W.pickle'):
        with open('tmp/W.pickle', 'rb') as W_file:
            W = pickle.load(W_file)
    else:
        print('Calculating the weight matrix...')
        W = np.zeros((num_vertices, num_vertices))
        for i in tqdm(range(num_vertices)):
            for j in range(num_vertices):
                if A[i, j] != 0:
                    w = (1/np.sum(np.square(mesh.vertices[i]-mesh.vertices[j]))) * A[i, j]
                    W[i, j] = w

        # Normalize W
        print('Normalizing...')
        for i in tqdm(range(num_vertices)):
            W[i, :] = W[i, :]/np.linalg.norm(W[i, :], 1)

        with open('tmp/W.pickle', 'wb') as W_file:
            pickle.dump(W, W_file)

    L = D - W

    #L = laplacian_calculation(mesh, equal_weight=False).toarray()
    # Scatterplot
    # L_plot = (L != 0)
    # plt.matshow(L_plot)
    # plt.savefig('Laplacian.png')
    # # plt.show()
    # print('loaded')

    # Decompose the Matrix
    print('Decomposing matrix...')
    if False:  #os.path.exists('tmp/Eigen.pickle'):
        with open('tmp/Eigen.pickle', 'rb') as eigen_file:
            decomposed_matrices = pickle.load(eigen_file)
    else:
        decomposed_matrices = np.linalg.eigh(L)

        with open('tmp/Eigen.pickle', 'wb') as eigen_file:
            pickle.dump(decomposed_matrices, eigen_file)

    # Plot Laplacian spectrum
    plt.plot([x[0] for x in enumerate(decomposed_matrices[0])], decomposed_matrices[0])
    plt.show()




