import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

############################ experiment_design.ipynb ###########################


# TODO: offload the code in `experiment_design.ipynb` to here.


########################### creating_categories.ipynb ##########################


def sample_unit_vector(dims=200):
    x = np.random.uniform(-1, 1, size=dims)
    return x / np.linalg.norm(x)


def _create_category_means(dist, dims=200):
    # this version returns the hypersphere point
    x = sample_unit_vector(dims)
    mu = dist / 2 * x
    return mu, -mu, x


def create_category_means(dist, dims=200):
    mu1, mu2, _ = _create_category_pairs(dist, dims)
    return mu1, mu2


def hypersphere_demo(dist, SD, plot_axis=True, plot_SD_wireframe=True,
                     plot_scatter=True, fig=None):

	# Matplotlib configuration
    fig = plt.gcf() if fig is None else fig
    ax = fig.add_subplot(111, projection='3d')

    # Make data
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 40)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1, alpha=0.3)
    
    # Create categories
    mu1, mu2, original = _create_category_means(dist, dims=3)
    original_points = np.vstack([original, -original]).T
    mu_points = np.vstack([mu1, mu2]).T

    # Plot category means
    ax.scatter(*mu1, color='purple', s=200)
    ax.scatter(*mu2, color='orange', s=200)
    
    # Plot the axis used to generate the category means
    if plot_axis:
        ax.scatter(*original_points, color='r', s=200)
        ax.plot(*original_points, color='r', linestyle='--')
    
    # Sample points according to the two categories and scatter them
    if plot_scatter:
        sample1 = np.random.normal(mu1, SD, size=(1000, 3)).T
        sample2 = np.random.normal(mu2, SD, size=(1000, 3)).T
        ax.scatter(*sample1, color='purple', alpha=0.1)
        ax.scatter(*sample2, color='orange', alpha=0.1)
    
    # Plot a 1-SD wireframe sphere around means
    if plot_SD_wireframe:
        ax.plot_wireframe(
        	SD * x + mu1[0], SD * y + mu1[1], SD * z + mu1[2],
        	color='purple', rstride=1, cstride=1, alpha=0.1)
        ax.plot_wireframe(
        	SD * x + mu2[0], SD * y + mu2[1], SD * z + mu2[2],
        	color='orange', rstride=1, cstride=1, alpha=0.1)