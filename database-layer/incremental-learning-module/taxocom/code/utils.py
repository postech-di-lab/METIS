import os
import math
import queue
import operator
import filenames
import numpy as np

def evaluate_cluster_specs(kappas, n_known_cluster, n_novel_cluster):
    if n_novel_cluster == 0: return True, np.inf
    if n_known_cluster == 0: return True, np.std(kappas)
    if np.inf in kappas: return False, np.inf
    
    known_kappas = kappas[:n_known_cluster]
    novel_kappas = kappas[n_known_cluster:]
    valid_flag = True
    for novel_kappa in novel_kappas:
        if np.max(known_kappas) < novel_kappa:
            valid_flag = False
        if np.min(known_kappas) > novel_kappa:
            valid_flag = False
    return valid_flag, np.std(kappas)

def load_embeddings(embedding_file):
    if embedding_file is None:
        return {}
    word_to_vec = {}
    with open(embedding_file, 'r') as fin:
        header = fin.readline()
        for line in fin:
            items = line.strip().split()
            word = items[0]
            vec = np.array([float(v) for v in items[1:]])
            word_to_vec[word] = vec
    return word_to_vec

def load_hierarchy(hier_file):
    hier_map = {}
    with open(hier_file) as f:
        idx = 0
        for line in f:
            topic = line.split()[0]
            hier_map[topic] = idx
            idx += 1
    return hier_map

def load_term_clusters(clus_file):
    clus_map = {}
    with open(clus_file) as f:
        for line in f:
            clus_id, ph, _ = line.strip('\r\n').split('\t')
            clus_id = int(clus_id)
            if clus_id not in clus_map:
                clus_map[clus_id] = []
            clus_map[clus_id].append(ph)
    return clus_map

def load_taxonomy(taxo_file):
    taxonomy = {}
    with open(taxo_file, 'r') as f:
        for line in f:
            segments = line.strip('\n\r ').split('\t')
            parent_node, child_nodes = segments[0], segments[1:]
            taxonomy[parent_node] = child_nodes
    return taxonomy  

def ensure_directory_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_all_subnodes(taxonomy, target_node):
    results, q = [target_node], queue.Queue()
    q.put(target_node)
    while not q.empty():
        node = q.get()
        for child_node in taxonomy.get(node, []):
            if child_node not in results:
                results.append(child_node)
                q.put(child_node)
    return results

def write_term_clusters(center_terms, term_clusters, term_scores, anchor_terms, node_dir):
    output_file = os.path.join(node_dir, filenames.term_clusters)
    with open(output_file, 'w') as fout:
        for clus_id in range(len(center_terms)):
            ranked_terms = sorted([(term, term_scores.get(term, 0.0)) for term in term_clusters[clus_id]], key=lambda x: -x[1])
            for term, score in ranked_terms:
                if term in anchor_terms:
                    fout.write(str(clus_id) + '\t' + term + '\t' + str(score) + '\n')
                    
    for clus_id, center_term in enumerate(center_terms):
        child_output_dir = os.path.join(node_dir, center_term)
        child_output_file = os.path.join(node_dir, center_term, filenames.seed_terms)
        ensure_directory_exist(child_output_dir)
        with open(child_output_file, 'w') as fout:
            for term in term_clusters[clus_id]:
                if term in anchor_terms:
                    fout.write(term + '\n')

def write_doc_clusters(center_terms, doc_clusters, node_dir):
    output_file = os.path.join(node_dir, filenames.doc_clusters)
    with open(output_file, 'w') as fout:
        for clus_id in range(len(center_terms)):
            doc_ids = doc_clusters[clus_id]
            for doc_id in doc_ids:
                fout.write(str(clus_id) + '\t' + str(doc_id) + '\n')

    for clus_id, center_term in enumerate(center_terms):
        child_output_dir = os.path.join(node_dir, center_term)
        child_output_file = os.path.join(node_dir, center_term, filenames.doc_ids)
        ensure_directory_exist(child_output_dir)
        doc_ids = doc_clusters[clus_id]
        with open(child_output_file, 'w') as fout:
            for doc_id in doc_ids:
                fout.write(str(doc_id) + '\n')

def write_term_scores(term_scores, anchor_terms, node_dir):
    output_score_file = os.path.join(node_dir, filenames.term_scores)
    with open(output_score_file, 'w') as fout:
        for term, score in term_scores:
            fout.write(term + '\t' + str(score) + '\n')

    output_term_file = os.path.join(node_dir, filenames.anchor_terms)
    with open(output_term_file, 'w') as fout:
        for term in anchor_terms:
            fout.write(term + '\n')

def write_center_terms(center_terms, node_name, node_dir):
    output_file = os.path.join(node_dir, filenames.hierarchy)
    with open(output_file, 'w') as fout:
        for center_term in center_terms:
            fout.write(center_term + '\t' + node_name + '\n')

def write_novelty_scores(threshold, novelty_scores, term_similarity, terms, node_dir):
    output_file = os.path.join(node_dir, filenames.novelty_scores)
    with open(output_file, 'w') as fout:
        fout.write('Threshold\t' + str(threshold) + '\n')
        for idx, novelty_score in enumerate(novelty_scores):
            fout.write(terms[idx] + '\t' + str(novelty_score) + '\n')
