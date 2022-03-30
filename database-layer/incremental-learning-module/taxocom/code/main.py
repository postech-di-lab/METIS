import os, time
import filenames
import argparse
from distutils.file_util import copy_file

from dataset import DataSet
from embedding import run_embedding
from clustering import run_clustering

MAX_LEVEL = 2

def recur(input_corpus, node_dir, node_name, filter_tau, n_locterms, betas, level):

    if level == MAX_LEVEL: return
    print('============================= Running level ', level, ' and node ', node_name, '=============================')

    start = time.time()
    run_embedding(input_corpus, n_locterms, node_dir, node_name)
    end = time.time()
    print("[Main] Finish locally discriminative embedding - %s seconds" % (end - start))

    start = time.time()
    child_names = run_clustering(input_corpus, filter_tau, betas, node_name, node_dir, level)
    end = time.time()
    print("[Main] Finish novelty adaptive clustering - %s seconds" % (end - start))
    
    for child_name in child_names:
        child_dir = os.path.join(node_dir, child_name)
        src_file = os.path.join(node_dir, filenames.embeddings)
        tgt_file = os.path.join(child_dir, filenames.embeddings)
        copy_file(src_file, tgt_file)
        recur(input_corpus, child_dir, child_name, filter_tau, n_locterms, betas, level+1)
        os.remove(tgt_file)

def main(args):
    input_dir = os.path.join(args.data_dir, args.dataset, 'input')
    root_dir = os.path.join(args.data_dir, args.dataset, 'root_' + args.seed_taxo)

    # read the entire input corpus
    start = time.time()
    corpus_file = os.path.join(input_dir, filenames.docs)
    index_file = os.path.join(input_dir, filenames.index)
    term_freq_file = os.path.join(input_dir, filenames.term_freq)
    term_integrity_file = os.path.join(input_dir, filenames.term_integrity)
    taxo_file = os.path.join(input_dir, args.seed_taxo+'.txt')
    input_corpus = DataSet(corpus_file, index_file, term_freq_file, taxo_file, term_integrity_file)
    end = time.time()
    print('[Main] Done reading the full data - %s seconds' % (end - start))

    # initialize the root directory
    if not os.path.exists(root_dir): os.makedirs(root_dir)
    copy_file(os.path.join(input_dir, filenames.doc_ids), os.path.join(root_dir, filenames.doc_ids))
    copy_file(os.path.join(input_dir, filenames.docs), os.path.join(root_dir, filenames.docs))
    copy_file(os.path.join(input_dir, filenames.terms), os.path.join(root_dir, filenames.seed_terms))
    copy_file(os.path.join(input_dir, filenames.embeddings), os.path.join(root_dir, filenames.embeddings))

    # start recursive discovery of novel topic clusters
    recur(input_corpus, root_dir, '*', args.filter_tau, args.n_locterms, args.betas, 0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/', type=str, help='path to the data directory')
    parser.add_argument('--dataset', default='nyt', type=str, help='name of the dataset')
    parser.add_argument('--seed_taxo', default='seed_taxo', type=str, help='name of the given taxonomy')
    parser.add_argument('--filter_tau', default=0.3, type=float, help='threshold for filtering out non-anchor terms')
    parser.add_argument('--n_locterms', default=100, type=int, help='number of relevant terms for retrieving docs')
    parser.add_argument('--betas', default=[1.8, 3.0], nargs='+', type=float, help='beta for the novelty threshold')
    args = parser.parse_args()

    main(args)

