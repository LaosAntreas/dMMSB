import matplotlib.pyplot as plt
import numpy as np


def plot_roles(true_vectors, inferred_vectors=None, title=None):
    """
    Plots role vectors on a simplex (K=3 only).
    
    Parameters:
    - true_vectors: Array of shape (N, K).
    - inferred_vectors: (Optional) Array of shape (N, K).
    - title: Optional plot title.
    """
    assert np.allclose(np.sum(true_vectors, axis=1), 1), \
        "True vectors should sum to 1 across roles."
    assert true_vectors.shape[1] == 3, \
        "This plotting function only supports K=3."

    if inferred_vectors is not None:
        assert np.allclose(np.sum(inferred_vectors, axis=1), 1), \
            "Inferred vectors should sum to 1 across roles."
        assert inferred_vectors.shape[1] == 3, \
            "This plotting function only supports K=3."

    # Triangle vertices
    v1, v2, v3 = np.array([0,0]), np.array([1,0]), np.array([0.5, np.sqrt(3)/2])

    def barycentric_to_cartesian(p):
        return p[0]*v1 + p[1]*v2 + p[2]*v3

    true_points = np.array([barycentric_to_cartesian(p) for p in true_vectors])

    plt.figure(figsize=(6,6))
    plt.scatter(true_points[:,0], true_points[:,1], s=20, marker='o',
                alpha=0.7, edgecolors='b', facecolors='none', label='Ground Truth')

    if inferred_vectors is not None:
        inferred_points = np.array([barycentric_to_cartesian(p) for p in inferred_vectors])
        plt.scatter(inferred_points[:,0], inferred_points[:,1], s=20, marker='x',
                    alpha=0.7, color='r', label='Inferred')
        for t, inf in zip(true_points, inferred_points):
            plt.plot([t[0], inf[0]], [t[1], inf[1]], '-', c='k', alpha=0.3)

    # Draw simplex outline
    plt.plot([v1[0], v2[0], v3[0], v1[0]],
             [v1[1], v2[1], v3[1], v1[1]], 'k--')

    plt.axis('off')
    plt.axis('equal')
    if title:
        plt.title(title)
    plt.legend()
    plt.show()