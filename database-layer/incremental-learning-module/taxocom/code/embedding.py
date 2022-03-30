import os
import subprocess
import utils
import filenames
import numpy as np
from dataset import SubDataSet

def retrieve_local_docs(node_name, dataset, n_expand, term_doc_map, taxonomy):
    relevant_terms = get_relevant_terms(node_name, dataset, n_expand)
    relevant_terms = relevant_terms.union(utils.get_all_subnodes(taxonomy, node_name))
    relevant_doc_ids = get_relevant_docs(relevant_terms, term_doc_map)
    relevant_doc_ids = relevant_doc_ids.union(dataset.doc_ids)
    return relevant_doc_ids

def get_relevant_terms(query, dataset, N):
    query_embedding = dataset.embeddings[query]
    relevant_term_scores = np.matmul(dataset.term_embeddings, query_embedding)
    relevant_term_ids = np.argsort(-relevant_term_scores)[:N]
    relevant_terms = set([dataset.terms[term_id] for term_id in relevant_term_ids])
    return relevant_terms

def get_relevant_docs(relevant_terms, term_doc_map):
    relevant_doc_ids = set()
    for term in relevant_terms:
        relevant_doc_ids = relevant_doc_ids.union(term_doc_map.get(term, []))
    return relevant_doc_ids

def train_josd(node_dir, node_name, docs, doc_ids, taxonomy):
    print('Starting cell %s with %d docs.' % (node_name, len(doc_ids)))
    
    input_f = os.path.join(node_dir, filenames.docs)
    cate_f = os.path.join(node_dir, filenames.child_names)

    with open(input_f, 'w') as f:
        for doc_id in doc_ids:
            f.write(' '.join(docs[doc_id]) + '\n')

    with open(cate_f, 'w') as f:
        for child_name in taxonomy.get(node_name, []):
            f.write(child_name + '\t')
            for seed_term in utils.get_all_subnodes(taxonomy, child_name):
                if seed_term == child_name: continue
                f.write(seed_term + '\t')
            f.write('\n')
    
    run_command = ['./josd', '-threads', '30', '-train', input_f, '-category-file', cate_f]
    run_command += ['-load-emb', os.path.join(node_dir, filenames.embeddings)]
    run_command += ['-word-emb', os.path.join(node_dir, filenames.word_embeddings)]
    run_command += ['-cate-emb', os.path.join(node_dir, filenames.center_embeddings)]

    subprocess.call(run_command)
    os.remove(input_f)

def run_embedding(input_corpus, n_expand, node_dir, node_name):

    doc_id_file = os.path.join(node_dir, filenames.doc_ids)
    term_file = os.path.join(node_dir, filenames.seed_terms)
    emb_file = os.path.join(node_dir,filenames.embeddings)

    dataset = SubDataSet(input_corpus, doc_id_file, term_file, emb_file)
    if node_name == '*' : # root node
        local_doc_ids = dataset.doc_ids
    else:
        local_doc_ids = retrieve_local_docs(node_name, dataset, n_expand, input_corpus.term_doc_map, input_corpus.taxonomy)

    train_josd(node_dir, node_name, input_corpus.docs, local_doc_ids, input_corpus.taxonomy)
    
