
source ./common.ini
overwrite=1

if [ ! -d "$rootpath" ]; then
    echo "The rootpath $rootpath does not exist"
    exit
fi


for synset_name in imagenet1k imagenet1k2hop
do
    if [ "$synset_name" == "imagenet1k" ]; then
        synsetfile=data/ilsvrc12/synsets.txt
    elif [ "$synset_name" == "imagenet1k2hop"  ]; then
        synsetfile=data/ilsvrc12/synsets2hop.txt
    else
        echo "unknown synset_name $synset_name"
        exit
    fi


    for embedding in conse conse2 hierse hierse2
    do
        python build_synset_vec.py $synsetfile $synset_name --embedding $embedding --word2vec $word2vec --corpus $corpus  --overwrite $overwrite
    done
done




