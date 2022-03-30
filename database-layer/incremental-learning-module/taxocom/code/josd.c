// The code structure (especially file reading and saving functions) is adapted from the Word2Vec implementation
//          https://github.com/tmikolov/word2vec

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>

#define MAX_STRING 100
#define MAX_WORDS_NODE 100
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const int corpus_max_size = 40000000;  // Maximum 40M documents in the corpus
const int topic_max_num = 1000;  // Maximum 1000 topics in the corpus

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

struct topic_node {
  // emb: node embedding
  real *emb, *wt_score, *grad;
  // *cur_words: array of vocabulary indices of current retrieved representative words
  // init_size: how many seed words are given
  // cur_size: total number of current retrieved representative words
  int node_id, *cur_words, init_size, cur_size, *relation_id, *siblings, num_siblings, level;
  int *children, num_children, parent;
  real margin;
  char *node_name;
};
    

char train_file[MAX_STRING], word_emb_file[MAX_STRING], document_emb_file[MAX_STRING], res_file[MAX_STRING];
char category_emb_file[MAX_STRING], matrix_file[MAX_STRING], level_file[MAX_STRING];
char save_vocab_file[MAX_STRING], load_emb_file[MAX_STRING], read_vocab_file[MAX_STRING], category_file[MAX_STRING];
struct vocab_word *vocab;
struct topic_node *topic_tree;
struct topic_node *topic_list;
long long *doc_sizes;
int binary = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 20, min_reduce = 1;
int num_per_topic = 10; // top-k words per topic to show
int *vocab_hash, *docs;
long long vocab_max_size = 1000, vocab_size = 0, corpus_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 10, pretrain_iter = 0, file_size = 0, iter_count;
real alpha = 0.025, starting_alpha, sample = 1e-3, global_lambda = 1.5, lambda_dis = 1.0, lambda_cat = 1.0;
real word_margin = 0.3, cat_margin = 0.9, dis_margin;
real *syn0, *syn1, *syn1neg, *syn1doc, *wt_score_ptr, *level_margin;
clock_t start;
int *rankings;

// how many words to pass before embedding treint dis_emb_period = 128;
int dis_emb_period = 128;
long long nodes, max_level = 0;

int negative = 2, expand = 1;
const int table_size = 1e8;
int *table;


void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

int IntCompare(const void * a, const void * b) { 
  return *(int*)a - *(int*)b; 
}

int SimCompare(const void *a, const void *b) { // large -> small
  return (wt_score_ptr[*(int *) a] < wt_score_ptr[*(int *) b]) - (wt_score_ptr[*(int *) a] > wt_score_ptr[*(int *) b]);
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *) "</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Locate line number of current file pointer
int FindLine(FILE *fin) {
  long long pos = ftell(fin);
  long long lo = 0, hi = corpus_size - 1;
  while (lo < hi) {
    long long mid = lo + (hi - lo) / 2;
    if (doc_sizes[mid] > pos) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  long long l = ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
  if (l > 0) return 1;
  if (l < 0) return -1;
  return 0;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i, wc = 0;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("[ERROR]: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    wc++;
    if ((debug_mode > 1) && (wc >= 1000000)) {
      printf("%lldM%c", train_words / 1000000, 13);
      fflush(stdout);
      wc = 0;
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    }
    else if (i == 0) {
      vocab[i].cn++;
      doc_sizes[corpus_size] = ftell(fin);
      corpus_size++;
      if (corpus_size >= corpus_max_size) {
        printf("[ERROR] Number of documents in corpus larger than \"corpus_max_size\"! Set a larger \"corpus_max_size\" in Line 20 of cate.c!\n");
        exit(1);
      }
    }
    else {
      vocab[i].cn++;
      docs[corpus_size]++;
    }
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("[ERROR]: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

// Update root node embedding
void UpdateRoot() {
  int i = 0, j, a;
  for (a = 0; a < layer1_size; a++) topic_tree[i].emb[a] = 0;
  for (j = 0; j < topic_tree[i].num_children; j++) {
    int child = topic_tree[i].children[j];
    for (a = 0; a < layer1_size; a++) topic_tree[i].emb[a] += topic_tree[child].emb[a];
  }
  real norm = 0.0;
  for (a = 0; a < layer1_size; a++) 
    norm += topic_tree[i].emb[a] * topic_tree[i].emb[a];
  for (a = 0; a < layer1_size; a++)
    topic_tree[i].emb[a] /= sqrt(norm);
}

void ReadCategoryName() {
  long long a, i = 0, j;
  int vocab_idx;
  //char tmp_word[MAX_STRING];
  char *tmp_word = NULL;
  real norm = 0.0;
  //memset(tmp_word, '\0', sizeof(tmp_word));

  // Read category name file
  FILE *f = fopen(category_file, "rb");
  printf("Category name file: %s\n", category_file);
  if (f == NULL) {
    printf("Category name not found\n");
    exit(1);
  }

  char *line = NULL;
  size_t len = 0;
  ssize_t read;

  // allocate and initialize topic_nodes
  topic_list = (struct topic_node *)calloc(topic_max_num, sizeof(struct topic_node));
  rankings = (int *)calloc(vocab_size, sizeof(int));
  for (a = 0; a < vocab_size; a++) rankings[a] = a;

  i = 0;
  while ((read = getline(&line, &len, f)) != -1) {
    topic_list[i].node_name = (char *)calloc(MAX_STRING, sizeof(char));
    topic_list[i].cur_words = (int *)calloc(MAX_WORDS_NODE, sizeof(int));
    topic_list[i].emb = (real *)calloc(layer1_size, sizeof(real));
    topic_list[i].init_size = 0;
    topic_list[i].grad = (real *)calloc(layer1_size, sizeof(real));
    topic_list[i].wt_score = (real *)calloc(vocab_size, sizeof(real));
    topic_list[i].node_id = i;

    line[read - 1] = 0;
    if (line[read - 2] == '\r') // windows line ending
      line[read - 2] = 0;

    tmp_word = strtok (line, "\t");
    strcpy(topic_list[i].node_name, tmp_word);
    printf("Target category %s : ", tmp_word);
    while (tmp_word != NULL) {
      if ((vocab_idx = SearchVocab(tmp_word)) != -1) {
        topic_list[i].cur_words[topic_list[i].init_size++] = vocab_idx;
        printf("%s ", tmp_word);
      } else {
        printf("[ERROR] Category name %s not found in vocabulary!\n", tmp_word);
        exit(1);
      }
      tmp_word = strtok(NULL, "\t");
    }
    printf("\n");

    topic_list[i].cur_size = topic_list[i].init_size;
    for (j = 0; j < topic_list[i].cur_size; j++) {
      int word = topic_list[i].cur_words[j];
      for (a = 0; a < layer1_size; a++) topic_list[i].emb[a] += syn0[a + word*layer1_size];
    }
    norm = 0.0;
    for (a = 0; a < layer1_size; a++) 
      norm += topic_list[i].emb[a] * topic_list[i].emb[a];
    for (a = 0; a < layer1_size; a++)
      topic_list[i].emb[a] /= sqrt(norm);
    i++;
  }
  fclose(f);
  nodes = i;
}

void LoadEmb(char *emb_file, real *emb_ptr) {
  long long a, b, c;
  int *vocab_match_tmp = (int *) calloc(vocab_size + 1, sizeof(int));
  int pretrain_vocab_size = 0, vocab_size_tmp = 0, word_dim;
  char *current_word = (char *) calloc(MAX_STRING, sizeof(char));
  real *syn_tmp = NULL, norm;
  unsigned long long next_random = 1;
  a = posix_memalign((void **) &syn_tmp, 128, (long long) layer1_size * sizeof(real));
  if (syn_tmp == NULL) {
    printf("[ERROR] Memory allocation failed\n");
    exit(1);
  }
  printf("Loading embedding from file %s\n", emb_file);
  if (access(emb_file, R_OK) == -1) {
    printf("[ERROR] File %s does not exist\n", emb_file);
    exit(1);
  }
  // read embedding file
  FILE *fp = fopen(emb_file, "r");
  fscanf(fp, "%d", &pretrain_vocab_size);
  fscanf(fp, "%d", &word_dim);
  if (layer1_size != word_dim) {
    printf("[ERROR] Embedding dimension incompatible with pretrained file!\n");
    exit(1);
  }

  vocab_size_tmp = 0;
  for (c = 0; c < pretrain_vocab_size; c++) {
    fscanf(fp, "%s", current_word);
    a = SearchVocab(current_word);
    if (a == -1) {
      for (b = 0; b < layer1_size; b++) fscanf(fp, "%f", &syn_tmp[b]);
    }
    else {
      norm = 0.0;
      for (b = 0; b < layer1_size; b++) {
        fscanf(fp, "%f", &emb_ptr[a * layer1_size + b]);
        norm += emb_ptr[a * layer1_size + b] * emb_ptr[a * layer1_size + b];
      }
      for (b = 0; b < layer1_size; b++)
        emb_ptr[a * layer1_size + b] /= sqrt(norm);
      vocab_match_tmp[vocab_size_tmp] = a;
      vocab_size_tmp++;
    }
  }
  // printf("In vocab: %d\n", vocab_size_tmp);
  qsort(&vocab_match_tmp[0], vocab_size_tmp, sizeof(int), IntCompare);
  vocab_match_tmp[vocab_size_tmp] = vocab_size;
  int i = 0;
  for (a = 0; a < vocab_size; a++) {
    if (a < vocab_match_tmp[i]) {
      norm = 0.0;
      for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        emb_ptr[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
        norm += emb_ptr[a * layer1_size + b] * emb_ptr[a * layer1_size + b];
      }
      for (b = 0; b < layer1_size; b++)
        emb_ptr[a * layer1_size + b] /= sqrt(norm);
    }
    else if (i < vocab_size_tmp) {
      i++;
    }
  }

  fclose(fp);
  free(current_word);
  free(emb_file);
  free(vocab_match_tmp);
  free(syn_tmp);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  real norm;
  a = posix_memalign((void **) &syn0, 128, (long long) vocab_size * layer1_size * sizeof(real));
  a = posix_memalign((void **) &syn1neg, 128, (long long) vocab_size * layer1_size * sizeof(real));
  a = posix_memalign((void **) &syn1doc, 128, (long long) corpus_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {
    printf("Memory allocation failed (syn0)\n");
    exit(1);
  }
  if (syn1neg == NULL) {
    printf("Memory allocation failed (syn1neg)\n");
    exit(1);
  }
  if (syn1doc == NULL) {
    printf("Memory allocation failed (syn1doc)\n");
    exit(1);
  }
  
  if (load_emb_file[0] != 0) {
    char *center_emb_file = (char *) calloc(MAX_STRING, sizeof(char));
    strcpy(center_emb_file, load_emb_file);
    LoadEmb(center_emb_file, syn0);
  }
  else {
    for (a = 0; a < vocab_size; a++) {
      norm = 0.0;
      for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
        norm += syn0[a * layer1_size + b] * syn0[a * layer1_size + b];
      }
      for (b = 0; b < layer1_size; b++)
        syn0[a * layer1_size + b] /= sqrt(norm);
    }
  }
  for (a = 0; a < vocab_size; a++) {
    norm = 0.0;
    for (b = 0; b < layer1_size; b++) {
      next_random = next_random * (unsigned long long) 25214903917 + 11;
      syn1neg[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
      norm += syn1neg[a * layer1_size + b] * syn1neg[a * layer1_size + b];
    }
    for (b = 0; b < layer1_size; b++)
      syn1neg[a * layer1_size + b] /= sqrt(norm);
  }
  for (a = 0; a < corpus_size; a++) {
    norm = 0.0;
    for (b = 0; b < layer1_size; b++) {
      next_random = next_random * (unsigned long long) 25214903917 + 11;
      syn1doc[a * layer1_size + b] = (((next_random & 0xFFFF) / (real) 65536) - 0.5) / layer1_size;
      norm += syn1doc[a * layer1_size + b] * syn1doc[a * layer1_size + b];
    }
    for (b = 0; b < layer1_size; b++)
      syn1doc[a * layer1_size + b] /= sqrt(norm);
  }

}

// update margin (m_inter) for each level
void UpdateMargin() {
  int u, v, c;
  real h;

  // zero out margins
  dis_margin = 0;
  int count = 0;

  for (u = 0; u < nodes; u++) {
    for (v = u+1; v < nodes; v++) {
      h = 0;
      for (c = 0; c < layer1_size; c++) h += topic_list[u].emb[c] * topic_list[v].emb[c];
      count += 1;
      dis_margin += h;
    }
  }
  if (count >= 1) dis_margin /= count;
}

real DisEmb() {
  int u, v, c, cnt = 0;
  real g, h, loss = 0, cur_margin = 0;
  real *grad = (real *)calloc(layer1_size, sizeof(real));

  if (nodes == 0 || nodes == 1) {
    free(grad);
    return loss;
  }    

  // zero out gradient
  for (u = 0; u < nodes; u++) 
    for (c = 0; c < layer1_size; c++) 
      topic_list[u].grad[c] = 0;

  for (u = 0; u < nodes; u++) {
    cur_margin = dis_margin;
    for (v = u+1; v < nodes; v++) {
      cnt++;
      h = 0;
      for (c = 0; c < layer1_size; c++) h += topic_list[u].emb[c] * topic_list[v].emb[c];

      if (h > cur_margin) {
        loss += h - cur_margin;
        
        // node u gradient
        for (c = 0; c < layer1_size; c++) grad[c] = h * topic_list[u].emb[c] - topic_list[v].emb[c];
        for (c = 0; c < layer1_size; c++) topic_list[u].grad[c] += alpha * lambda_dis / nodes * grad[c];

        // node v gradient
        for (c = 0; c < layer1_size; c++) grad[c] = h * topic_list[v].emb[c] - topic_list[u].emb[c];
        for (c = 0; c < layer1_size; c++) topic_list[v].grad[c] += alpha * lambda_dis / nodes * grad[c];
      }
    }
  }

  // update node embeddings
  for (u = 0; u < nodes; u++) {
    for (c = 0; c < layer1_size; c++)
      topic_list[u].emb[c] += topic_list[u].grad[c];
    g = 0;
    for (c = 0; c < layer1_size; c++) g += topic_list[u].emb[c] * topic_list[u].emb[c];
    for (c = 0; c < layer1_size; c++) topic_list[u].emb[c] /= sqrt(g);
  }

  free(grad);
  return loss / cnt;
}

// embed representative words close to corresponding category node
real CatEmb() {
  long long i, j, word, c, cnt = 0;
  real f, g, loss = 0;
  real *grad = (real *)calloc(layer1_size, sizeof(real));

  if (nodes == 0 || nodes == 1) {
    free(grad);
    return loss;
  } 

  for (i = 0; i < nodes; i++) {
    for (c = 0; c < layer1_size; c++) 
      topic_list[i].grad[c] = 0;
    for (j = 0; j < topic_list[i].cur_size; j++) {
      word = topic_list[i].cur_words[j];
      cnt++;
      f = 0;
      for (c = 0; c < layer1_size; c++) f += syn0[c + word*layer1_size] * topic_list[i].emb[c];
      if (f < cat_margin) {
        loss += cat_margin - f;
        for (c = 0; c < layer1_size; c++) grad[c] = syn0[c + word*layer1_size] - f * topic_list[i].emb[c];
        for (c = 0; c < layer1_size; c++) topic_list[i].grad[c] += alpha * lambda_cat / topic_list[i].cur_size * grad[c];
        
        for (c = 0; c < layer1_size; c++) grad[c] = topic_list[i].emb[c] - f * syn0[c + word*layer1_size];
        for (c = 0; c < layer1_size; c++) syn0[c + word*layer1_size] += alpha * lambda_cat / topic_list[i].cur_size * grad[c];
        g = 0;
        for (c = 0; c < layer1_size; c++) g += syn0[c + word*layer1_size] * syn0[c + word*layer1_size];
        for (c = 0; c < layer1_size; c++) syn0[c + word*layer1_size] /= sqrt(g);
      }
    }
    for (c = 0; c < layer1_size; c++) topic_list[i].emb[c] += topic_list[i].grad[c];
    g = 0;
    for (c = 0; c < layer1_size; c++) g += topic_list[i].emb[c] * topic_list[i].emb[c];
    for (c = 0; c < layer1_size; c++) topic_list[i].emb[c] /= sqrt(g);
  }
    
  free(grad);
  return loss / cnt;
}

void ExpandTopic() {
  long a, b, c;
  int cur_sz, flag;
  real norm;
  for (a = 0; a < nodes; a++) {
    for (b = 0; b < vocab_size; b++) {
      topic_list[a].wt_score[b] = 0;
      norm = 0.0;
      for (c = 0; c < layer1_size; c++) {
        topic_list[a].wt_score[b] += topic_list[a].emb[c] * syn0[b * layer1_size + c];
        norm += syn0[b * layer1_size + c] * syn0[b * layer1_size + c];
      }
      topic_list[a].wt_score[b] /= sqrt(norm);
    }
    wt_score_ptr = topic_list[a].wt_score;
    qsort(rankings, vocab_size, sizeof(int), SimCompare);
    cur_sz = topic_list[a].init_size;
    while (cur_sz < topic_list[a].cur_size + expand) {
      for (b = 0; b < vocab_size; b++) {
        flag = 0;
        for (c = 0; c < cur_sz; c++) {
          if (rankings[b] == topic_list[a].cur_words[c]) {
            flag = 1;
            break;
          }
        }
        if (flag == 0) {
          topic_list[a].cur_words[cur_sz++] = rankings[b];
          break;
        }
      }
    }
    topic_list[a].cur_size += expand;
  }
}

void *TrainModelThread(void *id) {
  long long a, b, d, word, doc = 0, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, l3 = 0, c, target, local_iter = 1;
  int word_counter = 0;
  unsigned long long next_random = (long long)id;
  real f, g, h, step, dis_loss = 0, cat_loss = 0;
  clock_t now;
  real *neu1 = (real *) calloc(layer1_size, sizeof(real));
  real *grad = (real *) calloc(layer1_size, sizeof(real));
  real *neu1e = (real *) calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Discrimination Loss: %f  Category Loss: %f  Progress: %.2f%%  Words/thread/sec: %.2fk", 
         13, alpha, dis_loss, cat_loss,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      if (global_lambda > 0) doc = FindLine(fi);
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    
    if (word_counter == dis_emb_period) word_counter = 0;
    if (word_counter == 0 && iter_count >= pretrain_iter) {
      dis_loss = DisEmb();
      cat_loss = CatEmb();
    }
    word_counter++;

    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    // b = 0;

    for (a = b; a < window * 2 + 1 - b; a++)
      if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size; // positive center word u
        
        // obj_w = 0;
        for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            l3 = word * layer1_size; // positive context word v
          } else {
            next_random = next_random * (unsigned long long) 25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            l2 = target * layer1_size; // negative center word u'
            f = 0;
            for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l3]; // f = cos(v, u) = v * u
            h = 0;
            for (c = 0; c < layer1_size; c++) h += syn0[c + l2] * syn1neg[c + l3]; // h = cos(v, u') = v * u'
        
            if (f - h < word_margin) {
              // obj_w += word_margin - (f - h);

              // compute context word gradient
              for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
              for (c = 0; c < layer1_size; c++) neu1e[c] += syn0[c + l1] - f * syn1neg[c + l3] + h * syn1neg[c + l3] - syn0[c + l2];
              
              // update positive center word
              for (c = 0; c < layer1_size; c++) grad[c] = syn1neg[c + l3] - f * syn0[c + l1]; // negative Riemannian gradient
              step = 1 - f; // cosine distance, d_cos
              for (c = 0; c < layer1_size; c++) syn0[c + l1] += alpha * step * grad[c];
              g = 0;
              for (c = 0; c < layer1_size; c++) g += syn0[c + l1] * syn0[c + l1];
              g = sqrt(g);
              for (c = 0; c < layer1_size; c++) syn0[c + l1] /= g;

              // update negative center word
              for (c = 0; c < layer1_size; c++) grad[c] = h * syn0[c + l2] - syn1neg[c + l3];
              step = 2 * h; // 2 * negative cosine similarity
              for (c = 0; c < layer1_size; c++) syn0[c + l2] += alpha * step * grad[c];
              g = 0;
              for (c = 0; c < layer1_size; c++) g += syn0[c + l2] * syn0[c + l2];
              g = sqrt(g);
              for (c = 0; c < layer1_size; c++) syn0[c + l2] /= g;

              // update context word
              step = 1 - (f - h);
              for (c = 0; c < layer1_size; c++) syn1neg[c + l3] += alpha * step * neu1e[c];
              g = 0;
              for (c = 0; c < layer1_size; c++) g += syn1neg[c + l3] * syn1neg[c + l3];
              g = sqrt(g);
              for (c = 0; c < layer1_size; c++) syn1neg[c + l3] /= g;
            }
          }
        }
      }

    // obj_d = 0;
    l1 = doc * layer1_size; // positive document d
    for (d = 0; d < negative + 1; d++) {
      if (d == 0) {
        l3 = word * layer1_size; // positive center word u
      } else {
        next_random = next_random * (unsigned long long) 25214903917 + 11;
        target = table[(next_random >> 16) % table_size];
        if (target == 0) target = next_random % (vocab_size - 1) + 1;
        if (target == word) continue;
        l2 = target * layer1_size; // negative center word u'
      
        f = 0;
        for (c = 0; c < layer1_size; c++) f += syn0[c + l3] * syn1doc[c + l1]; // f = cos(u, d) = u * d
        h = 0;
        for (c = 0; c < layer1_size; c++) h += syn0[c + l2] * syn1doc[c + l1]; // h = cos(u', d) = u' * d
    
        if (f - h < word_margin) {
          // obj_d += word_margin - (f - h);

          // compute document gradient
          for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
          for (c = 0; c < layer1_size; c++) neu1e[c] += syn0[c + l3] - f * syn1doc[c + l1] + h * syn1doc[c + l1] - syn0[c + l2];

          // update positive center word
          for (c = 0; c < layer1_size; c++) grad[c] = syn1doc[c + l1] - f * syn0[c + l3];
          step = 1 - f;
          for (c = 0; c < layer1_size; c++) syn0[c + l3] += alpha * step * grad[c];
          g = 0;
          for (c = 0; c < layer1_size; c++) g += syn0[c + l3] * syn0[c + l3];
          g = sqrt(g);
          for (c = 0; c < layer1_size; c++) syn0[c + l3] /= g;

          // update negative center word
          for (c = 0; c < layer1_size; c++) grad[c] = h * syn0[c + l2] - syn1doc[c + l1];
          step = 2 * h;
          for (c = 0; c < layer1_size; c++) syn0[c + l2] += alpha * step * grad[c];
          g = 0;
          for (c = 0; c < layer1_size; c++) g += syn0[c + l2] * syn0[c + l2];
          g = sqrt(g);
          for (c = 0; c < layer1_size; c++) syn0[c + l2] /= g;

          // update document
          step = 1 - (f - h);
          for (c = 0; c < layer1_size; c++) syn1doc[c + l1] += alpha * step * neu1e[c];
          g = 0;
          for (c = 0; c < layer1_size; c++) g += syn1doc[c + l1] * syn1doc[c + l1];
          g = sqrt(g);
          for (c = 0; c < layer1_size; c++) syn1doc[c + l1] /= g;
        }
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  free(grad);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (word_emb_file[0] == 0) return;
  InitNet();
  InitUnigramTable();
  start = clock();

  ReadCategoryName();
  //printf("Pre-training for %lld epochs, in total %lld epochs\n", pretrain_iter, iter);
  for (iter_count = 0; iter_count < iter; iter_count++) {
    if (iter_count >= pretrain_iter) UpdateMargin();
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *) a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    if (iter_count >= pretrain_iter) ExpandTopic();
  }
  printf("\n");

  fo = fopen(word_emb_file, "wb");
  // Save the word vectors
  fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
  for (a = 0; a < vocab_size; a++) {
    fprintf(fo, "%s ", vocab[a].word);
    if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
    else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);

  fo = fopen(category_emb_file, "wb");
  // Save the tree embedding vectors
  fprintf(fo, "%lld %lld\n", nodes, layer1_size);
  for (a = 0; a < nodes; a++) {
    fprintf(fo, "%s ", topic_list[a].node_name);
    if (binary) for (b = 0; b < layer1_size; b++) fwrite(&topic_list[a].emb[b], sizeof(real), 1, fo);
    else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", topic_list[a].emb[b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
  
}

void WriteResult() {
  long a, b, c;
  int cur_sz, flag;
  real norm;
  FILE *fo = fopen(res_file, "wb");
  //printf("Topic mining results written to file %s\n", res_file);
  for (a = 0; a < nodes; a++) {
    for (b = 0; b < vocab_size; b++) {
      topic_list[a].wt_score[b] = 0;
      norm = 0;
      for (c = 0; c < layer1_size; c++) {
        topic_list[a].wt_score[b] += topic_list[a].emb[c] * syn0[b * layer1_size + c];
        norm += syn0[b * layer1_size + c] * syn0[b * layer1_size + c];
      }
      topic_list[a].wt_score[b] /= sqrt(norm);
    }
    wt_score_ptr = topic_list[a].wt_score;

    qsort(rankings, vocab_size, sizeof(int), SimCompare);

    fprintf(fo, "Category (%s):\n", topic_list[a].node_name);
    cur_sz = 0;
    for (b = 0; b < vocab_size; b++) {
      flag = 0;
      for (c = 0; c < topic_list[a].init_size; c++) {
        if (rankings[b] == topic_list[a].cur_words[c]) {
          flag = 1;
          break;
        }
      }
      if (flag == 0) {
        fprintf(fo, "%s ", vocab[rankings[b]].word);
        cur_sz++;
      }
      if (cur_sz >= num_per_topic) break;
    }
    fprintf(fo, "\n");
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("Parameters:\n");

    printf("\t##########   Input/Output:   ##########\n");
    printf("\t-train <file> (mandatory argument)\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-category-file <file>\n");
    printf("\t\tUse <file> to provide the topic names/keywords\n");
    printf("\t-matrix-file <file>\n");
    printf("\t\tUse <file> to provide the taxonomy file in matrix form; generated by read_taxo.py\n");
    printf("\t-level-file <file>\n");
    printf("\t\tUse <file> to provide the node level information file; generated by read_taxo.py\n");
    printf("\t-res <file>\n");
    printf("\t\tUse <file> to save the hierarchical topic mining results\n");
    printf("\t-k <int>\n");
    printf("\t\tSet the number of terms per topic in the output file; default is 10\n");
    printf("\t-word-emb <file>\n");
    printf("\t\tUse <file> to save the resulting word embeddings\n");
    printf("\t-tree-emb <file>\n");
    printf("\t\tUse <file> to save the resulting category embeddings\n");
    printf("\t-load-emb <file>\n");
    printf("\t\tThe pretrained embeddings will be read from <file>\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    

    printf("\n\t##########   Embedding Training:   ##########\n");
    printf("\t-size <int>\n");
    printf("\t\tSet dimension of text embeddings; default is 100\n");
    printf("\t-iter <int>\n");
    printf("\t\tSet the number of iterations to train on the corpus (performing topic mining); default is 5\n");
    printf("\t-pretrain <int>\n");
    printf("\t\tSet the number of iterations to pretrain on the corpus (without performing topic mining); default is 2\n");
    printf("\t-expand <int>\n");
    printf("\t\tSet the number of terms to be added per topic per iteration; default is 1\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-word-margin <float>\n");
    printf("\t\tSet the word embedding learning margin; default is 0.25\n");
    printf("\t-cat-margin <float>\n");
    printf("\t\tSet the intra-category coherence margin m_intra; default is 0.9\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 2, common values are 3 - 5 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    
    printf("\nSee run.sh for an example to set the arguments\n");
    
    return 0;
  }
  word_emb_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  res_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-category-file", argc, argv)) > 0) strcpy(category_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-res", argc, argv)) > 0) strcpy(res_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-k", argc, argv)) > 0) num_per_topic = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-matrix-file", argc, argv)) > 0) strcpy(matrix_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-level-file", argc, argv)) > 0) strcpy(level_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-load-emb", argc, argv)) > 0) strcpy(load_emb_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-global-lambda", argc, argv)) > 0) global_lambda = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda-dis", argc, argv)) > 0) lambda_dis = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-lambda-cat", argc, argv)) > 0) lambda_cat = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-word-margin", argc, argv)) > 0) word_margin = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-cat-margin", argc, argv)) > 0) cat_margin = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-word-emb", argc, argv)) > 0) strcpy(word_emb_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-docu-emb", argc, argv)) > 0) strcpy(document_emb_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-cate-emb", argc, argv)) > 0) strcpy(category_emb_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-disc-period", argc, argv)) > 0) dis_emb_period = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-expand", argc, argv)) > 0) expand = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-pretrain", argc, argv)) > 0) pretrain_iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  docs = (int *) calloc(corpus_max_size, sizeof(int));
  doc_sizes = (long long *)calloc(corpus_max_size, sizeof(long long));
  TrainModel();
  //WriteResult();
  return 0;
}
