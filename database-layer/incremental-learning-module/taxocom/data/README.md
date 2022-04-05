
## Prepare your own dataset

**Step 1. Place the files `docs.txt` and `terms.txt` in the directory `<dataset-name>/raw/`**

- `docs.txt`: Each row contains tokenized texts, which correspond to a single document
- `terms.txt`: Each row contains a candidate term that is potentially included in your output topic taxonomy

**Step 2. Run the preprocessing codes at the working directory `../code/`** 

```
bash run_preprocss.sh <dataset-name>
```

- It first pre-trains term embedding vectors by using your own corpus
  - Before you run the codes, please ensure that there exists the directory `<dataset-name>/input/`
  - It creates the initial embedding file `embeddings.txt` in the directory `<dataset-name>/input/`
- Then, it outputs several files required for executing the TaxoCom framework by preprocessing your files
- (Optional) You can leverage term integrity scores by placing the file `term_integrity.txt` in `<dataset-name>/input`

**Step 3. Place the file `<seed-taxo-name>.txt` in the directory `<dataset-name>/input/`**

- This file provides information of the initial topic structure (i.e., a partial hierarchy of topic names)
- Each row lists a parent node (i.e., a topic) and its child nodes (i.e., sub-topics), separated by a tab
```
* <fisrt-level-topic-1> <first-level-topic-2> ...
<first-level-topic-1>  <second-level-topic-1a> <second-level-topic-1b> ...
<first-level-topic-2>  <second-level-topic-2a> <second-level-topic-2b>  ...
...
```

## Useful tools for text mining

- In the paper, [AutoPhrase](https://github.com/shangjingbo1226/AutoPhrase) is used to tokenize the documents and to extract term integrity scores
