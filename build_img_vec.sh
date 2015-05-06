source ./common.ini

if [ ! -d "$rootpath" ]; then
    echo "The rootpath $rootpath does not exist"
    exit
fi

label_file=data/ilsvrc12/synsets.txt
synset_name=imagenet1k
label2vec_dir=$rootpath/synset2vec/$synset_name/$corpus,$word2vec,$embedding

testCollection=imagenet2hop-random2k
subset=$testCollection
new_feature=$corpus,$word2vec,$embedding,$feature
overwrite=1

python im2vec.py $label_file $label2vec_dir $testCollection $feature $new_feature --subset $subset --overwrite $overwrite


