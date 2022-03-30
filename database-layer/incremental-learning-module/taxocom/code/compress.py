import argparse
import utils
import operator
import queue
import math
import os
import filenames

def recursion(root_dir, output_file, N):
    q = queue.Queue()
    q.put((root_dir, '*'))
    g = open(output_file, 'w+')

    while not q.empty():
        (c_folder, c_name) = q.get()
        
        clus_f = os.path.join(c_folder, filenames.term_clusters)
        hier_f = os.path.join(c_folder, filenames.hierarchy)
        if not os.path.exists(hier_f): continue

        hier_map = utils.load_hierarchy(hier_f)
        clus_map = utils.load_term_clusters(clus_f)

        for cluster, clus_id in hier_map.items():
            cluster_folder = os.path.join(c_folder, cluster)
            cluster_namespace = c_name + '/' + cluster
            q.put((cluster_folder, cluster_namespace))

            terms = clus_map.get(clus_id, [])[:N]
            terms_str = ','.join(terms)
            g.write('%s\t%s\n' % (cluster_namespace, terms_str))

    g.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/', type=str, help='path to the data directory')
    parser.add_argument('--dataset', default='nyt', type=str, help='name of the dataset')
    parser.add_argument('--seed_taxo', default='seed_taxo', type=str, help='name of the given taxonomy')
    parser.add_argument('--N', default=10, type=int, help='number of terms included.')
    args = parser.parse_args()

    root_dir = os.path.join(args.data_dir, args.dataset, 'root_' + args.seed_taxo)
    output_file = '%s_%s_output.txt' % (args.dataset, args.seed_taxo)

    recursion(root_dir, output_file, args.N)


