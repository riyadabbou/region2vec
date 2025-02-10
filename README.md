# region2vec-GAT

**GeoAI-Enhanced Community Detection on Spatial Networks with Graph Deep Learning**
 
region2vec: Community Detection on Spatial Networks Using Graph Embedding with Node Attributes and Spatial Interactions

![Region2vec](https://github.com/GeoDS/region2vec-GAT/blob/master/Region2Vec_Workflow.jpg)
![Region2vec](https://github.com/GeoDS/region2vec-GAT/blob/master/Region2Vec_results.jpg)

**Abstract:** 
Spatial networks are essential for modeling geographic phenomena where spatial interactions play a significant role. To analyze these networks and their internal structures, graph-based techniques like community detection are commonly employed. Community detection aims to identify strongly connected components within the network and uncover hidden relationships between nodes. However, traditional methods often overlook node attribute information.

To address this limitation, this study introduces region2vec, a family of GeoAI-enhanced, unsupervised community detection methods built on Graph Attention Networks (GAT) and Graph Convolutional Networks (GCN). Region2vec generates neural embeddings for nodes by integrating attribute similarity, geographic proximity, and spatial interactions. These embeddings are then used to detect network communities through agglomerative clustering.

The proposed GeoAI-based methods are benchmarked against several baseline approaches and demonstrate superior performance, particularly in maximizing both node attribute similarity and spatial interaction intensity within spatial network communities. Additionally, their application to public health, specifically in identifying shortage areas, highlights their potential in addressing regionalization challenges.


