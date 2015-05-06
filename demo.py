import os

testCollection = 'imagenet2hop-random2k'

for embedding in str.split('conse conse2 hierse hierse2'):
    method_name = 'flickr4m,tagvec500,%s/dascaffeprob' % embedding
    os.system('python zero_shot_tagging.py %s --embedding %s --overwrite 1' % (testCollection, embedding))
    os.system('python eval.py %s %s' % (testCollection, method_name))


