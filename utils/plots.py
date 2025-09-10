import matplotlib.pyplot as plt
import numpy as np


def plot_roles(true_vectors, inferred_vectors, title=None):
    """
    Plots the true and inferred role vectors for comparison.

    Parameters:
    - true_vectors: Array of shape (N, K) representing the true role vectors.
    - inferred_vectors: Array of shape (N, K) representing the inferred role vectors.
    - title: Optional title for the plot.
    """
    assert np.allclose(np.sum(true_vectors, axis=1), 1), "True vectors should sum to 1 across roles."
    assert np.allclose(np.sum(inferred_vectors, axis=1), 1), "Inferred vectors should sum to 1 across roles."
    assert true_vectors.shape[1] == 3 and inferred_vectors.shape[1]==3, "This plotting function only supports K=3."

    N, K = true_vectors.shape
    # Triangle vertices (equilateral triangle)
    v1 = np.array([0, 0])
    v2 = np.array([1, 0])
    v3 = np.array([0.5, np.sqrt(3)/2])

    # Barycentric -> Cartesian
    def barycentric_to_cartesian(p):
        return p[0]*v1 + p[1]*v2 + p[2]*v3

    true_points = np.array([barycentric_to_cartesian(p) for p in true_vectors])

    inferred_points = np.array([barycentric_to_cartesian(p) for p in inferred_vectors])



    # Plot
    plt.figure(figsize=(6,6))
    plt.scatter(true_points[:,0], true_points[:,1], s=10, marker='o', alpha=0.5, facecolors='none', edgecolors='b')
    plt.scatter(inferred_points[:,0], inferred_points[:,1], s=10, marker='x', alpha=0.5, color='r')
    for i in range(N):
        plt.plot([true_points[i,0], inferred_points[i,0]], [true_points[i,1], inferred_points[i,1]], '-', c='k', alpha=0.3)
    plt.plot([v1[0], v2[0], v3[0], v1[0]],
            [v1[1], v2[1], v3[1], v1[1]], 'k--')  # Triangle outline
    plt.axis('equal')
    if title:
        plt.title(title)
    plt.show()
        