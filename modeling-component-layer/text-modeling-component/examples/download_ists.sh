wget http://alt.qcri.org/semeval2016/task2/data/uploads/test_goldstandard.tar.gz

mkdir -p data/ISTS
tar xvf test_goldstandard.tar.gz
mv test_goldstandard.tar.gz data/ISTS
mv test_goldStandard data/ISTS