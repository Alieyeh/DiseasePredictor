import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


class DiseasePredictor:
    def __init__(self):
        self.HSDN_matrix = None
        self.HSDN_labels = None

    def calculate_matrix_of_HSDN(self, HSDN_path):
        g = nx.read_edgelist(HSDN_path, delimiter='\t', create_using=nx.DiGraph(), data=(('weight', np.float),))
        self.HSDN_labels = np.array(list(g.nodes))
        self.HSDN_matrix = nx.to_numpy_array(g, nodelist=self.HSDN_labels, weight='weight').transpose((1, 0))

    def __get_symptom_vector(self, symptoms):
        vector = np.zeros((1, np.shape(self.HSDN_labels)[0]))
        for symptom, level in symptoms.items():
            level_int = ['very-low', 'med', 'high', 'very-high'].index(level)
            vector[0, np.where(self.HSDN_labels == symptom)] = level_int / 3
        return vector

    def __get_prob_of_diseases(self, symptoms):
        vector = self.__get_symptom_vector(symptoms)
        sim = cosine_similarity(self.HSDN_matrix, vector)
        sim = np.reshape(sim, newshape=self.HSDN_labels.shape[0])
        prob = sim / np.sum(sim)
        return prob

    def __get_prob_of_diseases_v2(self, symptoms):
        vector = self.__get_symptom_vector(symptoms)
        sim = np.sum(self.HSDN_matrix * vector, axis=1)
        sum = np.sum(sim)
        if sum == 0:
            sum = 1
        prob = sim / sum
        return prob

    def get_k_top_related_disease(self, symptoms, k):
        prob_vector = self.__get_prob_of_diseases_v2(symptoms)
        sorted_k_args = np.flip(np.argsort(prob_vector))[:k]
        diseases = []
        for i in sorted_k_args:
            if prob_vector[i] != 0:
                diseases.append((self.HSDN_labels[i], prob_vector[i] * 100))
        return diseases

    def get_k_top_related_symptoms(self, symptoms, k):
        symptoms_vector = self.__get_symptom_vector(symptoms)
        p_vector = self.__get_prob_of_diseases_v2(symptoms)

        allowed_features_indices = np.where(symptoms_vector == 0)[1]
        clean_matrix = self.HSDN_matrix[allowed_features_indices, :]
        clean_matrix = clean_matrix[:, allowed_features_indices]
        clean_labels = self.HSDN_labels[allowed_features_indices]
        clean_p_vector = p_vector[allowed_features_indices]

        coef_vector = (clean_p_vector * 10000).astype(int)
        replicated_matrix = np.repeat(clean_matrix, coef_vector, axis=0)
        replicated_labels = np.repeat(clean_labels, coef_vector, axis=0)

        model = SelectKBest(chi2, k).fit(replicated_matrix, replicated_labels)
        mask = model.get_support()
        new_features = [feature for bool_e, feature in zip(mask, clean_labels) if bool_e]
        return new_features
