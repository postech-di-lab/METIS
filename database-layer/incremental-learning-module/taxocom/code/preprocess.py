import os, sys
import filenames
import argparse
from collections import Counter

def trim_terms(raw_term_file, term_file, embedding_file):
    terms = load_terms(raw_term_file)
    embedded_terms = load_embedding_terms(embedding_file)
    with open(term_file, 'w') as fout:
        for w in terms:
            if w in embedded_terms:
                fout.write(w + '\n')

def load_terms(seed_word_file):
    seed_words = []
    with open(seed_word_file, 'r') as fin:
        for line in fin:
            seed_words.append(line.strip())
    return seed_words

def load_embedding_terms(embedding_file):
    term_set = set()
    with open(embedding_file, 'r') as fin:
        header = fin.readline()
        for line in fin:
            items = line.strip().split()
            word = items[0]
            term_set.add(word)
    return term_set

def trim_document_set(raw_doc_file, raw_doc_label_file, doc_file, doc_label_file, term_file, doc_label_flag=True):
    term_set = set(load_terms(term_file))

    if doc_label_flag:
        with open(raw_doc_file, 'r') as fin, open(doc_file, 'w') as fout, \
                open(raw_doc_label_file, 'r') as fin_, open(doc_label_file, 'w') as fout_:
            for line, line_ in zip(fin, fin_):
                doc, doc_label = line.strip().split(), line_.strip()
                if check_doc_contain_term(doc, term_set):
                    fout.write(' '.join(doc) + '\n')
                    fout_.write(doc_label + '\n')
    else:
        with open(raw_doc_file, 'r') as fin, open(doc_file, 'w') as fout:
            for line in fin:
                doc = line.strip().split()
                if check_doc_contain_term(doc, term_set):
                    fout.write(' '.join(doc) + '\n')

def check_doc_contain_term(doc, term_set):
    for word in doc:
        if word in term_set:
            return True
    return False

def gen_doc_term_cnt_file(doc_file, term_cnt_file):
    documents = []
    with open(doc_file, 'r') as fin:
        for line in fin:
            terms = line.strip().split()
            documents.append(terms)
    doc_word_counts = []
    for d in documents:
        c = Counter(d)
        doc_word_counts.append(c)

    with open(term_cnt_file, 'w') as fout:
        for i, counter in enumerate(doc_word_counts):
            counter_string = counter_to_string(counter)
            fout.write(str(i) + '\t' + counter_string + '\n')

def counter_to_string(counter):
    elements = []
    for k, v in counter.items():
        elements.append(k)
        elements.append(v)
    return '\t'.join([str(e) for e in elements])

def gen_doc_ids(input_file, output_file):
    doc_id = 0
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            fout.write(str(doc_id)+"\n")
            doc_id += 1

def build_index_file(doc_file, term_file, index_file):
    candidates = []
    with open(term_file) as f:
      for line in f:
        candidates.append(line.strip('\r\n'))

    pd_map = {x:set() for x in candidates}
    candidates_set = set(candidates)

    with open(doc_file) as f:
      idx = 0
      for line in f:
        tokens = line.strip('\r\n').split(' ')
        for t in tokens:
          if t in candidates_set:
            pd_map[t].add(str(idx))
        idx += 1

    with open(index_file, 'w+') as g:
      for ph in pd_map:
        if len(pd_map[ph]) > 0:
          doc_str = ','.join(pd_map[ph])
        else:
          doc_str = ''
        g.write('%s\t%s\n' % (ph, doc_str))

def main(raw_dir, input_dir):
    ## Following are three required input files
    raw_doc_file = os.path.join(raw_dir, filenames.docs)
    raw_doc_label_file = os.path.join(raw_dir, filenames.doc_labels)
    raw_term_file = os.path.join(raw_dir, filenames.terms)
    emb_file = os.path.join(input_dir, filenames.embeddings)

    ## Following are four output files
    doc_file = os.path.join(input_dir, filenames.docs)
    doc_label_file = os.path.join(input_dir, filenames.doc_labels)
    term_file = os.path.join(input_dir, filenames.terms)
    doc_term_cnt_file = os.path.join(input_dir, filenames.term_freq)
    doc_id_file = os.path.join(input_dir, filenames.doc_ids)
    index_file = os.path.join(input_dir, filenames.index)

    trim_terms(raw_term_file, term_file, emb_file)
    print('Done trimming the terms.')

    trim_document_set(raw_doc_file, raw_doc_label_file, doc_file, doc_label_file, term_file, doc_label_flag=False)
    print('Done trimming the documents.')

    gen_doc_term_cnt_file(doc_file, doc_term_cnt_file)
    print('Done counting the terms in documents.')

    gen_doc_ids(doc_file, doc_id_file)
    print('Done generating the doc ids.')

    build_index_file(doc_file, term_file, index_file)
    print('Done building the inverted index.')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/', type=str, help='path to the data directory')
    parser.add_argument('--dataset', default='nyt', type=str, help='name of the dataset')
    args = parser.parse_args()

    raw_dir = os.path.join(args.data_dir, args.dataset, 'raw')
    input_dir = os.path.join(args.data_dir, args.dataset, 'input')
    main(raw_dir, input_dir)
