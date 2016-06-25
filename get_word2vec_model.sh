
source common.ini

if [ ! -d "$rootpath" ]; then
    echo "rootpath $rootpath does not exit"
    exit
fi


DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

#wget http://www.mmc.ruc.edu.cn/research/hierse/flickr4m.word2vec.tar.gz
wget http://lixirong.net/data/sigir2015/flickr4m.word2vec.tar.gz
# or manually download the zipped file from https://drive.google.com/open?id=0B89Vll9z5OVEfnRHUWRSY0dkRjNuRVZYUGtzY0ltVTZ2bkRvSVBTRjd0akEwckVMZGV6WTQ&authuser=0

echo "Unzipping..."

tar -xf flickr4m.word2vec.tar.gz -C $rootpath && rm -f flickr4m.word2vec.tar.gz

echo "Done."

