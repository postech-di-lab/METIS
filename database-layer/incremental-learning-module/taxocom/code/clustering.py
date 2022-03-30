from collections import defaultdict
from scipy.spatial.distance import cosine
from spherecluster import SphericalKMeans, VonMisesFisherMixture

from dataset import SubDataSet
from case_slim import run_caseolap
from utils import evaluate_cluster_specs
from utils import write_novelty_scores, write_center_terms
from utils import write_term_clusters, write_doc_clusters

import os
import numpy as np
import filenames

class Clusterer:
    def __init__(self, data, n_cluster):
        self.data = data
        self.n_cluster = n_cluster
        self.clus = SphericalKMeans(n_cluster)
        self.assignment = None  # a list contain the membership of the data points
        self.confidence = None  # a list contain the confidence of the data points
        self.center_ids = None  # a list contain the ids of the cluster centers
        self.inertia_scores = None

    def fit(self):
        self.clus.fit(self.data)
        self.assignment = self.clus.labels_
        self.center_ids = self.generate_center_idx()
        self.confidence = self.compute_confidence()
        self.inertia_scores = self.clus.inertia_
        return self

    def compute_confidence(self):
        centers = self.clus.cluster_centers_
        membership = np.matmul(self.data, centers.T)
        confidence = np.max(membership, axis=1)
        return confidence

    def generate_center_idx(self):
        ret = []
        for cluster_id in range(self.n_cluster):
            center_idx = self.find_center_idx_for_one_cluster(cluster_id)
            ret.append(center_idx)
        return ret

    def find_center_idx_for_one_cluster(self, cluster_id):
        query_vec = self.clus.cluster_centers_[cluster_id]
        members = np.where(self.assignment == cluster_id)[0]
        best_similarity, ret = -1, -1
        for member_idx in members:
            member_vec = self.data[member_idx]
            cosine_sim = self.calc_cosine(query_vec, member_vec)
            if cosine_sim > best_similarity:
                best_similarity = cosine_sim
                ret = member_idx
        return ret

    def calc_cosine(self, vec_a, vec_b):
        return 1 - cosine(vec_a, vec_b)


def run_clustering(input_corpus, filter_thre, betas, node_name, node_dir, level):

    doc_id_file = os.path.join(node_dir, filenames.doc_ids)
    seed_term_file = os.path.join(node_dir, filenames.seed_terms)
    emb_file = os.path.join(node_dir, filenames.embeddings)
    cemb_file = os.path.join(node_dir, filenames.center_embeddings)

    dataset = SubDataSet(input_corpus, doc_id_file, seed_term_file, emb_file, cemb_file)
    known_center_terms = [term for term in input_corpus.taxonomy.get(node_name, []) if term in dataset.terms]
    n_known_cluster = len(known_center_terms)

    # STEP1. known-topic / novel-topic term discrimination
    novel_term_identification(dataset, node_dir, beta=betas[level])

    # STEP2-1. known-topic term cluster assignment
    if n_known_cluster == 0:
        known_clusters, known_scores = {}, {}
    else:
        known_clusters, known_scores = known_cluster_assignment(dataset)

    best_discrepancy = None
    n_novel_clusters = [2, 3, 4] if n_known_cluster == 0 else [1, 2, 3] # ranges for the nyt dataset
    for n_novel_cluster in n_novel_clusters:
        # STEP2-2. novel-topic term cluster assignment
        if n_novel_cluster == 0 or len(dataset.novel_terms) == 0:
            novel_clusters, novel_scores, novel_center_terms = {}, {}, []
        else:
            novel_clusters, novel_scores, novel_center_terms = novel_cluster_detection(dataset, n_novel_cluster)

        # STEP2-3. combine term clustering results
        term_clusters = {}
        term_clusters.update(known_clusters)
        term_clusters.update({n_known_cluster + clus_id: clus_terms for clus_id, clus_terms in novel_clusters.items()})

        term_rel_scores = {}
        term_rel_scores.update(known_scores)
        term_rel_scores.update(novel_scores)

        # STEP3-1. document cluster assignment
        doc_clusters = dataset.get_doc_clusters(term_clusters, term_rel_scores)
        term_rep_scores = run_caseolap(term_clusters, doc_clusters, dataset.terms, input_corpus.term_integrity, input_corpus.term_freq_map)
        
        # STEP3-2. anchor term selection
        term_scores, anchor_terms = {}, []
        for term in dataset.terms:
            term_score = term_rel_scores.get(term, 0.0) * term_rep_scores.get(term, 0.0)
            if term_rep_scores.get(term, 0.0) > filter_thre:
                anchor_terms.append(term)
                term_scores[term] = term_score

        # STEP4-1. movmf estimation
        n_cluster_terms, kappas = movmf_cluster_estimation(dataset, term_clusters, anchor_terms)

        # STEP4-2. compute stdev for (Kc + Kc*) number of kappas
        valid_flag, discrepancy = evaluate_cluster_specs(kappas, n_known_cluster, n_novel_cluster)
        if best_discrepancy is None or best_discrepancy > discrepancy:
            best_discrepancy = discrepancy
            best_n_novel_cluster = n_novel_cluster
            best_clustering_result = [novel_center_terms, term_clusters, term_scores, anchor_terms]

        center_terms = known_center_terms + novel_center_terms
        result = '%d-known, %d-novel =>' % (n_known_cluster, n_novel_cluster)
        result += ' stdev=%.2f' % discrepancy
        for center_term, n_cluster_term, kappa in zip(center_terms, n_cluster_terms, kappas):
            result +=  ' ("%s": N=%d, k=%.2f)' % (center_term, n_cluster_term, kappa)
        print(result)

    n_novel_cluster = best_n_novel_cluster
    novel_center_terms, term_clusters, term_scores, anchor_terms = best_clustering_result
    center_terms = known_center_terms + novel_center_terms

    term_clusters = {clus_id: [term for term in terms if term in anchor_terms] for clus_id, terms in term_clusters.items()}
    doc_clusters = dataset.get_doc_clusters(term_clusters, term_scores)

    # STEP5. write the clustering results and create the child node directories
    write_term_clusters(center_terms, term_clusters, term_scores, anchor_terms, node_dir)
    write_doc_clusters(center_terms, doc_clusters, node_dir)
    write_center_terms(center_terms, node_name, node_dir)

    return center_terms

def novel_term_identification(dataset, node_dir, temperature=0.1, beta=1.5):
    # in case of newly-inserted nodes
    if len(dataset.center_embeddings) == 0: 
        dataset.known_terms = []
        dataset.novel_terms = dataset.terms
        return

    centers = dataset.center_embeddings
    term_similarity = np.matmul(dataset.term_embeddings, centers.T)
    if centers.shape[0] == 1: # sigmoid case : need to be modified (threshold)
        threshold = 0.5
        novelty_scores = 1 - 1/(1 + np.exp(-term_similarity / temperature))
    else: # softmax case
        threshold = (1 - 1/centers.shape[0]) ** beta 
        term_membership = np.exp(term_similarity / temperature)
        term_membership /= np.sum(term_membership, axis=1, keepdims=True)
        novelty_scores = 1 - np.max(term_membership, axis=1)
    write_novelty_scores(threshold, novelty_scores, term_similarity, dataset.terms, node_dir)

    known_terms, novel_terms = [], []
    for term_id, novelty_score in enumerate(novelty_scores):
        term = dataset.terms[term_id]
        if novelty_score < threshold: 
            known_terms.append(term)
        else:
            novel_terms.append(term)
    dataset.known_terms = known_terms
    dataset.novel_terms = novel_terms

def known_cluster_assignment(dataset):
    centers = dataset.center_embeddings

    term_membership = np.matmul(dataset.term_embeddings, centers.T)
    term_assignment = np.argmax(term_membership, axis=1)
    term_confidence = np.max(term_membership, axis=1)

    term_scores = {}
    term_clusters = {clus_id: [] for clus_id in range(len(centers))}
    for term_id, (clus_id, conf) in enumerate(zip(term_assignment, term_confidence)):
        term = dataset.terms[term_id]
        if term in dataset.known_terms:
            term_clusters[clus_id].append(term)
            term_scores[term] = conf
    return term_clusters, term_scores

def novel_cluster_detection(dataset, n_cluster):
    novel_term_ids = [dataset.term_to_id[novel_term] for novel_term in dataset.novel_terms]
    novel_term_embeddings = dataset.term_embeddings[novel_term_ids]
    clus = Clusterer(novel_term_embeddings, n_cluster).fit()

    term_scores = {}
    term_clusters = {clus_id: [] for clus_id in range(n_cluster)}
    for term_id, (clus_id, conf) in enumerate(zip(clus.assignment, clus.confidence)):
        term_clusters[clus_id].append(dataset.novel_terms[term_id])
        term_scores[dataset.novel_terms[term_id]] = conf

    center_terms = [dataset.novel_terms[center_id] for center_id in clus.center_ids]
    return term_clusters, term_scores, center_terms

def movmf_cluster_estimation(dataset, term_clusters, anchor_terms):
    n_cluster_terms, kappas = [], []
    for clus_id, cluster_terms in term_clusters.items():
        anchor_term_ids = [dataset.term_to_id[term] for term in cluster_terms if term in anchor_terms]
        anchor_term_embeddings = dataset.term_embeddings[anchor_term_ids]
        n_cluster_terms.append(len(anchor_term_ids))
        
        if len(anchor_term_ids) > 0:
            movmf = VonMisesFisherMixture(n_clusters=1, n_jobs=15, random_state=0)
            movmf.fit(anchor_term_embeddings)
            kappas.append(movmf.concentrations_)
        else:
            kappas.append([np.inf])

    kappas = np.concatenate(kappas)
    return n_cluster_terms, kappas
