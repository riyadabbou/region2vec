# region2vec-GAT

**GeoAI-Enhanced Community Detection on Spatial Networks with Graph Deep Learning**
 
region2vec: Community Detection on Spatial Networks Using Graph Embedding with Node Attributes and Spatial Interactions

![Region2vec](https://github.com/GeoDS/region2vec-GAT/blob/master/Region2Vec_Workflow.jpg)
![Region2vec](https://github.com/GeoDS/region2vec-GAT/blob/master/Region2Vec_results.jpg)

**Abstract:** 
Spatial networks are useful for modeling geographic phenomena where spatial interaction plays an important role. To analyze the spatial networks and their internal structures, graph-based methods such as community detection have been widely used. Community detection aims to extract strongly connected components from the network and reveal the hidden relationships between nodes, but they usually do not involve the attribute information. To consider edge-based interactions and node attributes together, this study proposed a family of GeoAI-enhanced unsupervised community detection methods called *region2vec* based on Graph Attention Networks (GAT) and Graph Convolutional Networks (GCN). The *region2vec* methods generate node neural embeddings based on attribute similarity, geographic adjacency and spatial interactions, and then extract network communities based on node embeddings using agglomerative clustering. The proposed GeoAI-based methods are compared with multiple baselines and perform the best when one wants to maximize node attribute similarity and spatial interaction intensity simultaneously within the spatial network communities. It is further applied in the shortage area delineation problem in public health and demonstrates its promise in regionalization problems. 


