import os
import math
import numpy as np
from collections import defaultdict

import filenames

# Entire input corpus
class DataSet:
    def __init__(self, doc_file, index_file, term_freq_file, taxo_file, integrity_file):
        self.docs = self.load_docs(doc_file)
        self.term_doc_map = self.load_index(index_file)
        self.term_freq_map = self.load_term_freq(term_freq_file)
        self.term_integrity = self.load_term_integrity(integrity_file)
        self.taxonomy = self.load_taxonomy(taxo_file)
        self.center_terms = [term for terms in self.taxonomy.values() for term in terms]
        
    def load_docs(self, doc_file):
        docs = []
        with open(doc_file, 'r') as fin:
            for line in fin:
                terms = line.strip().split()
                docs.append(terms)
        return docs
    
    def load_index(self, index_file):
        term_doc_map = {}
        with open(index_file, 'r') as f:
            for line in f:
                term, doc_ids = line.strip('\r\n').split('\t')
                doc_ids = doc_ids.split(',')
                if len(doc_ids) > 0 and doc_ids[0] == '':
                    continue
                term_doc_map[term] = set([int(x) for x in doc_ids])
        return term_doc_map

    def load_term_integrity(self, integrity_file):
        if not os.path.isfile(integrity_file): return None
        term_integrity = {}
        with open(integrity_file, 'r') as f:
            for line in f:
                term, integrity_score = line.split('\t')
                if term in self.term_doc_map:
                    term_integrity[term] = float(integrity_score)
        return term_integrity

    def load_term_freq(self, term_freq_file):
        term_freq_map = {}
        with open(term_freq_file, 'r') as f:
            for line in f:
                segments = line.strip('\n\r ').split('\t')
                doc_id = int(segments[0])
                term_freq_map[doc_id] = {}
                for i in range(1, len(segments), 2):
                    term, w = segments[i], int(segments[i+1])
                    term_freq_map[doc_id][term] = w
        return term_freq_map        

    def load_taxonomy(self, taxo_file):
        taxonomy = {}
        with open(taxo_file, 'r') as f:
            for line in f:
                segments = line.strip('\n\r ').split('\t')
                parent_node, child_nodes = segments[0], segments[1:]
                taxonomy[parent_node] = child_nodes
        return taxonomy     

# Sub-corpus for each topic node
class SubDataSet:
    def __init__(self, input_corpus, doc_id_file, term_file, emb_file, cemb_file=''):
        self.embeddings = self.load_raw_embeddings(emb_file)
        
        self.terms = self.load_terms(term_file)
        self.term_to_id = self.gen_term_id()
        self.term_set = set(self.terms)
        self.term_embeddings = self.get_term_embeddings()

        self.docs, self.doc_ids = self.load_documents(input_corpus, doc_id_file)
        self.center_embeddings = self.load_embeddings(cemb_file)
        
        self.term_idf = self.build_term_idf()

    def load_terms(self, term_file):
        terms = []
        with open(term_file, 'r') as fin:
            for line in fin:
                term = line.strip()
                if term in self.embeddings:
                    terms.append(term)
        return terms

    def gen_term_id(self):
        term_to_id = {}
        for idx, term in enumerate(self.terms):
            term_to_id[term] = idx
        return term_to_id

    def load_raw_embeddings(self, emb_file):
        term_to_vec = {}
        with open(emb_file, 'r', encoding='utf8', errors='ignore') as fin:
            header = fin.readline()
            for line in fin:
                items = line.strip().split()
                word = items[0]
                vec = [float(v) for v in items[1:]]
                term_to_vec[word] = vec
        return term_to_vec

    def load_embeddings(self, emb_file):
        if not os.path.isfile(emb_file): return []
        embeddings = []
        with open(emb_file, 'rb') as fin:
            header = fin.readline()
            for line in fin:
                items = line.strip().split()
                vec = [float(v) for v in items[1:]]
                embeddings.append(vec)
            if len(embeddings) > 0:
                embeddings = np.array(embeddings)
                embeddings /= np.sqrt(np.sum(embeddings ** 2, axis=1, keepdims=True))
        return embeddings

    def load_documents(self, input_corpus, doc_id_file):
        trimmed_doc_ids, trimmed_docs = [], []
        doc_ids = self.load_doc_ids(doc_id_file)
        for doc_id in doc_ids:
            doc = input_corpus.docs[doc_id]
            trimmed_doc = [e for e in doc if e in self.term_set]
            if len(trimmed_doc) > 0:
                trimmed_doc_ids.append(doc_id)
                trimmed_docs.append(trimmed_doc)
        return trimmed_docs, trimmed_doc_ids

    def load_doc_ids(self, doc_id_file):
        doc_ids = []
        with open(doc_id_file, 'r') as fin:
            for line in fin:
                doc_id = int(line.strip())
                doc_ids.append(doc_id)
        return doc_ids

    def get_term_embeddings(self):
        term_embeddings = []
        for word in self.terms:
            term_embeddings.append(self.embeddings[word])
        term_embeddings = np.array(term_embeddings)
        term_embeddings /= np.sqrt(np.sum(term_embeddings ** 2, axis=1, keepdims=True))
        return term_embeddings

    def build_term_idf(self):
        term_idf = defaultdict(float)
        for doc in self.docs:
            word_set = set(doc)
            for word in word_set:
                if word in self.term_set:
                    term_idf[word] += 1.0
        N = len(self.docs)
        for w in term_idf:
            term_idf[w] = math.log(1.0 + N / term_idf[w])
        return term_idf

    def get_doc_assignment(self, n_cluster, document, term_assignment, term_scores):
        doc_membership = [0.0] * n_cluster
        for term in document:
            term_id = self.term_to_id[term]
            clus_id = term_assignment[term_id]
            if clus_id is not None:
                idf = self.term_idf[term]
                rel = term_scores[term]
                doc_membership[clus_id] += idf * rel
        doc_assignment = doc_membership.index(max(doc_membership))
        return doc_assignment

    def get_doc_clusters(self, term_clusters, term_scores):
        term_assignment = [None] * len(self.term_to_id)
        for clus_id, terms in term_clusters.items():
            for term in terms:
                term_id = self.term_to_id[term]
                term_assignment[term_id] = clus_id
        n_cluster = len(term_clusters)
        doc_clusters = defaultdict(list)
        for idx, doc in zip(self.doc_ids, self.docs):
            doc_assignment = self.get_doc_assignment(n_cluster, doc, term_assignment, term_scores)
            doc_clusters[doc_assignment].append(idx)
        return doc_clusters

