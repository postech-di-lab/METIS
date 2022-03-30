DATA=$1
SEED_TAXO=$2
ROOT=root_$SEED_TAXO

echo 'Compile the discriminative embedding module'
gcc josd.c -o josd -lm -pthread -O2 -Wall -funroll-loops -Wno-unused-result

echo 'Create root folder for taxonomy'
if [ ! -d ../data/$DATA/$ROOT ]; then
	mkdir ../data/$DATA/$ROOT
fi
rm -rf ../data/$DATA/$ROOT/*

echo 'Run TaxoCom'
export PYTHONWARNINGS="ignore"
python -u main.py --dataset $DATA --seed_taxo $SEED_TAXO

echo 'Compress the output taxonomy'
python -u compress.py --dataset $DATA --seed_taxo $SEED_TAXO --N 10
