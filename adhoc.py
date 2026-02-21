import networkx as nx
from node2vec import Node2Vec
import numpy as np

# Create a graph
graph = nx.read_edgelist('resources/weighted_graph_dist.tsv', delimiter='\t', create_using=nx.DiGraph(), data=(('weight', np.double),))

# Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
node2vec = Node2Vec(graph, dimensions=100, walk_length=30, num_walks=100, workers=4)  # Use temp_folder for big graphs

# Embed nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)


# Save embeddings for later use
model.wv.save_word2vec_format('outputs/word2vec.csv')

# Save model for later use
model.save('outputs/word2vec_model')

# Look for most similar nodes
print(model.wv.most_similar('Irritable_Bowel_Syndrome'))  # Output node names are always strings



# import numpy as np
# from sklearn.metrics.pairwise import euclidean_distances
#
# array = np.loadtxt('outputs/word2vec_delLabel.csv')
# labels = np.loadtxt('outputs/word2vec_labels.csv', dtype=str)
# # print(array)
# # print(labels)
#
# dist = euclidean_distances(array, array)
# print(dist)
# print(len(np.where(dist < 0.02)[0]) - 4292)
# x = labels[np.where(dist < 0.02)[0]]
# y = labels[np.where(dist < 0.02)[1]]
#
# xy = np.array([x, y]).transpose()
#
# xy_neq = xy[np.where(xy[:, 0] != xy[:, 1])]
#
# print(xy_neq)
