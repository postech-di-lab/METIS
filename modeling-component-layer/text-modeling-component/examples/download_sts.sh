# Copyright (c) 2021, Seonghyeon Lee
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Download and tokenize data with MOSES tokenizer
data_path=data
preprocess_exec=./tokenizer.sed

# Get MOSES
echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git
SCRIPTS=mosesdecoder/scripts
MTOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LOWER=$SCRIPTS/tokenizer/lowercase.perl

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

PTBTOKENIZER="sed -f tokenizer.sed"

mkdir $data_path

SICK='http://alt.qcri.org/semeval2014/task1/data/uploads'
STSBenchmark='http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz'

# STS array ordered by 2012, 2013, 2014, 2015, 2016
STSNames=("STS12" "STS13" "STS14" "STS15" "STS16")
STSTasks=("MSRpar MSRvid SMTeuroparl surprise.OnWN surprise.SMTnews" "FNWN headlines OnWN" "deft-forum deft-news headlines OnWN images tweet-news" "answers-forums answers-students belief headlines images" "answer-answer headlines plagiarism postediting question-question")

STSPaths=("http://ixa2.si.ehu.es/stswiki/images/4/40/STS2012-en-test.zip" "http://ixa2.si.ehu.es/stswiki/images/2/2f/STS2013-en-test.zip" "http://ixa2.si.ehu.es/stswiki/images/8/8c/STS2014-en-test.zip" "http://ixa2.si.ehu.es/stswiki/images/d/da/STS2015-en-test.zip" "http://ixa2.si.ehu.es/stswiki/images/9/98/STS2016-en-test.zip")

STSSubdirs=("test-gold" "test-gs" "sts-en-test-gs-2014" "test_evaluation_task2a" "sts2016-english-with-gs-v1.0")

### STS datasets
# STS12, STS13, STS14, STS15, STS16
mkdir $data_path/STS

for index in ${!STSNames[@]}; do
  name=${STSNames[${index}]}
  tasks=${STSTasks[${index}]}
  fpath=${STSPaths[${index}]}
  subdir=${STSSubdirs[${index}]}

  echo $fpath
  curl -Lo $data_path/STS/data_$name.zip $fpath
  unzip $data_path/STS/data_$name.zip -d $data_path/STS
  mv $data_path/STS/${subdir} $data_path/STS/$name-en-test
  rm $data_path/STS/data_$task.zip

  for sts_task in ${tasks}; do
    fname=STS.input.$sts_task.txt
    task_path=$data_path/STS/$name-en-test/
    if [ "$name" = "STS16" ] ; then
      # Rename file
      echo 'Handling STS2016'
      mv $task_path/STS2016.input.$sts_task.txt $task_path/$fname
      mv $task_path/STS2016.gs.$sts_task.txt $task_path/STS.gs.$sts_task.txt
    fi
    # Apply mosestokenizer
    cut -f1 $task_path/$fname | $MTOKENIZER -threads 8 -l en -no-escape | $LOWER > $task_path/tmp1
    cut -f2 $task_path/$fname | $MTOKENIZER -threads 8 -l en -no-escape | $LOWER > $task_path/tmp2
    paste $task_path/tmp1 $task_path/tmp2 > $task_path/$fname
    rm $task_path/tmp1 $task_path/tmp2
  done
done

# STSBenchmark (http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark)
curl -Lo $data_path/Stsbenchmark.tar.gz $STSBenchmark
tar -zxvf $data_path/Stsbenchmark.tar.gz -C $data_path
rm $data_path/Stsbenchmark.tar.gz
mv $data_path/stsbenchmark $data_path/STS/STSBenchmark

for split in train dev test
do
    fname=sts-$split.csv
    fdir=$data_path/STS/STSBenchmark
    cut -f1,2,3,4,5 $fdir/$fname > $fdir/tmp1
    cut -f6 $fdir/$fname | $MTOKENIZER -threads 8 -l en -no-escape | $LOWER > $fdir/tmp2
    cut -f7 $fdir/$fname | $MTOKENIZER -threads 8 -l en -no-escape | $LOWER > $fdir/tmp3
    paste $fdir/tmp1 $fdir/tmp2 $fdir/tmp3 > $fdir/$fname
    rm $fdir/tmp1 $fdir/tmp2 $fdir/tmp3
done

### download SICK
mkdir $data_path/SICK

for split in train trial test_annotated
do
    urlname=$SICK/sick_$split.zip
    curl -Lo $data_path/SICK/sick_$split.zip $urlname
    unzip $data_path/SICK/sick_$split.zip -d $data_path/SICK/
    rm $data_path/SICK/readme.txt
    rm $data_path/SICK/sick_$split.zip
done

for split in train trial test_annotated
do
    fname=$data_path/SICK/SICK_$split.txt
    cut -f1 $fname | sed '1d' > $data_path/SICK/tmp1
    cut -f4,5 $fname | sed '1d' > $data_path/SICK/tmp45
    cut -f2 $fname | sed '1d' | $MTOKENIZER -threads 8 -l en -no-escape > $data_path/SICK/tmp2
    cut -f3 $fname | sed '1d' | $MTOKENIZER -threads 8 -l en -no-escape > $data_path/SICK/tmp3
    head -n 1 $fname > $data_path/SICK/tmp0
    paste $data_path/SICK/tmp1 $data_path/SICK/tmp2 $data_path/SICK/tmp3 $data_path/SICK/tmp45 >> $data_path/SICK/tmp0
    mv $data_path/SICK/tmp0 $fname
    rm $data_path/SICK/tmp*
done

# remove moses folder
rm -rf mosesdecoder
