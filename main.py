import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def get_input_symptoms(path, labels):
    symptoms = []
    with open(path) as f:
        data = f.read()
        for row in data.split('\n'):
            elements = row.split('\t')
            if len(elements) == 2:
                symptoms.append((elements[0], float(elements[1]) / 5))

    vector = np.zeros((1, np.shape(labels)[0]))
    for symptom, level in symptoms:
        vector[0, np.where(labels == symptom)] = level
    return vector


def get_matrix_of_HSDN(path):
    g = nx.read_edgelist(path, delimiter='\t', create_using=nx.DiGraph(), data=(('weight', np.double),))
    # edges = nx.read_edgelist('graph.tsv', data=(('similarity', float),),create_using=nx.DiGraph())
    # nx.draw(g, with_labels=True)
    # plt.show()
    # print(g.number_of_edges())
    # print(g.edges(data=True))
    # print(list(nx.neighbors(g,'Constipation')))
    labels = np.array(list(g.nodes))
    matrix = nx.to_numpy_array(g, nodelist=labels, weight='weight')
    matrix = matrix.transpose((1, 0))
    return matrix, labels


def get_prob_of_diseases(matrix, vector):
    sim = cosine_similarity(matrix, vector)
    sim = np.reshape(sim, newshape=HSDN_labels.shape[0])
    prob = sim / np.sum(sim)
    return prob


def get_k_top_related_symptoms(p_vector, labels, matrix, allowed_features_indices, k):
    clean_matrix = matrix[allowed_features_indices, :]
    clean_matrix = clean_matrix[:, allowed_features_indices]
    clean_labels = labels[allowed_features_indices]
    clean_p_vector = p_vector[allowed_features_indices]

    coef_vector = (clean_p_vector * 10000).astype(int)
    replicated_matrix = np.repeat(clean_matrix, coef_vector, axis=0)
    replicated_labels = np.repeat(clean_labels, coef_vector, axis=0)

    model = SelectKBest(chi2, k).fit(replicated_matrix, replicated_labels)
    mask = model.get_support()  # list of booleans
    new_features = [feature for bool_e, feature in zip(mask, clean_labels) if bool_e]
    return new_features


input_path = 'resources/user_inputs.txt'
graph_path = 'resources/weighted_graph.tsv'

HSDN_matrix, HSDN_labels = get_matrix_of_HSDN(graph_path)

# print(np.shape(HSDN_matrix))
# print(HSDN_matrix)
#
# print(np.shape(HSDN_labels))
# print(HSDN_labels)

symptoms_vector = get_input_symptoms(input_path, HSDN_labels)

# print(np.shape(symptoms_vector))
# print(symptoms_vector)

prob_vector = get_prob_of_diseases(matrix=HSDN_matrix, vector=symptoms_vector)

# np.savetxt('sim.csv', prob_vector)
# print(np.where(prob_vector > 0))
# sorted_args = np.flip(np.argsort(prob_vector))
# print(sorted_args)
# sorted_labels = HSDN_labels[sorted_args]
# sorted_prob = prob_vector[sorted_args]
#
# sorted_rslt = np.array([sorted_labels, sorted_prob]).transpose()
#
# np.savetxt('sortedLabels.tsv', sorted_rslt, fmt="%s", delimiter='\t')


k_top_related_symptoms = get_k_top_related_symptoms(
    prob_vector,
    HSDN_labels,
    HSDN_matrix,
    allowed_features_indices=np.where(symptoms_vector == 0)[1],
    k=3)

print(k_top_related_symptoms)
