
source common.ini

if [ ! -d "$rootpath" ]; then
    echo "rootpath $rootpath does not exit"
    exit
fi


DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

wget http://www.mmc.ruc.edu.cn/research/hierse/flickr4m.word2vec.tar.gz

echo "Unzipping..."

tar -xf flickr4m.word2vec.tar.gz -C $rootpath && rm -f flickr4m.word2vec.tar.gz

echo "Done."

