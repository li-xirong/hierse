
testCollection=imagenet2hop


for method in zeroshot.vecEmed_caffe1000Vote-Top10predict.ConSE2_vec200flickr25m zeroshot.vecEmed_caffe1000Vote-Top10predict.HierSE_vec200train1m zeroshot.vecEmed_caffe1000Vote-Top10predict.HierSE2_vec200flickr25m
do
    python eval.py $testCollection $method
done

