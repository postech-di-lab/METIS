DATA=$1

echo 'Compile the embedding module'
gcc jose.c -o jose -lm -pthread -O2 -Wall -funroll-loops -Wno-unused-result

echo 'Pretrain the term embeddings from the raw documents'
./jose -threads 30 -train ../data/$DATA/raw/docs.txt -word-emb ../data/$DATA/input/embeddings.txt

echo 'Preprocess the raw documents and terms'
python -u preprocess.py --dataset $DATA
