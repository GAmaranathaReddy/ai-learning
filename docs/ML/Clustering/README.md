# Clustering Algorithms

In real world, data does not always come with labels and in such cases, we need to create our own clusters/labels in order to make some groupings in the data. This is called unsupervised learning. The following algorithms are used to do this:

# Clustering Algorithms and Their Features

| Algorithm                                                                          | Category       | Features                                                                                                               |
| ---------------------------------------------------------------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------- |
| [K-Means](./K-meansClustering.md)                                                  | Partitioning   | - Simple and fast<br>- Works well with large datasets<br>- Assumes clusters are spherical and balanced                 |
| [Hierarchical Clustering](./HierarchicalClustering.md)                             | Hierarchical   | - Does not require the number of clusters to be specified<br>- Can be visualized using dendrograms                     |
| [DBSCAN (Density-Based)](./DBSCAN.md)                                              | Density-Based  | - Can find arbitrarily shaped clusters<br>- Good for data with noise and outliers                                      |
| [Mean Shift](./MeanShift.md)                                                       | Centroid-Based | - Does not require the number of clusters to be specified<br>- Can find clusters of any shape                          |
| [OPTICS (Ordering Points To Identify the Clustering Structure)](./OPTICS.md)       | Density-Based  | - Generalizes DBSCAN by addressing varying density clusters<br>- Provides a reachability plot for cluster hierarchy    |
| [Spectral Clustering](./Spectral.md)                                               | Graph-Based    | - Good for non-convex clusters<br>- Uses graph distance to cluster points                                              |
| [Affinity Propagation](./affinity-propagation.md)                                  | Graph-Based    | - Does not require the number of clusters to be specified<br>- Sends messages between pairs of samples                 |
| [Agglomerative Clustering](./agglomerative.md)                                     | Hierarchical   | - A type of hierarchical clustering<br>- Uses a bottom-up approach                                                     |
| [BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)](./birch.md) | Hierarchical   | - Designed for very large datasets<br>- Builds a tree-like structure to cluster                                        |

Note that this is not an exhaustive list, and there are many other clustering algorithms and variations thereof. Each algorithm has its own set of parameters and assumptions that can affect its performance on different datasets.
