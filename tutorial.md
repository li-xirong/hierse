
# Tutorial code for zero-shot image tagging


```python
from synset2vec import Synset2Vec
from im2vec import Image2Vec
from tagger import ZeroshotTagger
from simpleknn.bigfile import BigFile
from constant import ROOT_PATH as rootpath
```


```python
%run -i build_label_vec.py ilsvrc12_test1k
%run -i build_label_vec.py ilsvrc12_test1k_2hop
```

    [21 Aug 15:45:50 - build_label_vec.py:line 30] /Users/xirong/VisualSearch/synset2vec/ilsvrc12_test1k/flickr4m,tagvec500,hierse2/feature.bin exists. quit
    [21 Aug 15:45:50 - build_label_vec.py:line 30] /Users/xirong/VisualSearch/synset2vec/ilsvrc12_test1k_2hop/flickr4m,tagvec500,hierse2/feature.bin exists. quit



```python
# load image / label embedding models
i2v = Image2Vec()
s2v = Synset2Vec()
tagger = ZeroshotTagger()
```

    [21 Aug 15:45:50 - bigfile.py:line 24] 1000x500 instances loaded from /Users/xirong/VisualSearch/synset2vec/ilsvrc12_test1k/flickr4m,tagvec500,hierse2
    [21 Aug 15:45:50 - im2vec.py:line 44] #active_labels=1000, embedding_size=500
    [21 Aug 15:45:50 - bigfile.py:line 24] 382298x500 instances loaded from /Users/xirong/VisualSearch/flickr4m/word2vec/tagvec500
    [21 Aug 15:45:51 - synset2vec.py:line 27] w2v(flickr4m): 382298 words, 500 dims
    [21 Aug 15:45:51 - bigfile.py:line 24] 1548x500 instances loaded from /Users/xirong/VisualSearch/synset2vec/ilsvrc12_test1k_2hop/flickr4m,tagvec500,hierse2
    [21 Aug 15:45:51 - tagger.py:line 43] #active_labels=1548, embedding_size=500



```python
# get prediction scores for the known label set Y0, which is currently ilsvrc12_test1k
# In the following example we use socres computed in advance.
# Alternatively,call a pre-trained CNN model to get the scores on the fly 
image_collection = 'imagenet2hop-random2k'
test_image_id = 'n01495006_2522'
pY0 = 'dascaffeprob'
feat_dir = os.path.join(rootpath, image_collection, 'FeatureData', pY0)
feat_file = BigFile(feat_dir)
score_vec = feat_file.read_one(test_image_id) # 
assert (len(score_vec) == 1000) 
```

    [21 Aug 15:45:51 - bigfile.py:line 24] 2000x1000 instances loaded from /Users/xirong/VisualSearch/imagenet2hop-random2k/FeatureData/dascaffeprob



```python
# perform zero-shot image tagging
img_embedding_vec = i2v.embedding(score_vec)
res = tagger.predict(img_embedding_vec, topk=5)
print ([(label, s2v.explain(label), score) for (label,score) in res])
```

    [('n01482330', 'shark', 0.95676452140690293), ('n01488918', 'requiem shark', 0.95037136849784298), ('n01495006', 'shovelhead, bonnethead, bonnet shark, Sphyrna tiburo', 0.9460143660060516), ('n01483522', 'mackerel shark', 0.9248807358373593), ('n01494882', 'smalleye hammerhead, Sphyrna tudes', 0.90178226535080397)]

