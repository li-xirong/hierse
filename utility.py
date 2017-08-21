import os

from constant import *


def makedirsforfile(filename):
    if not os.path.exists(os.path.split(filename)[0]):
        os.makedirs(os.path.split(filename)[0])

def makedirs(dirname):
    if not os.path.exits(dirname):
        os.makedirs(dirname)

def readImageSet(testCollection, testset=None, rootpath=ROOT_PATH):
    if not testset:
        testset = testCollection
    imsetfile = os.path.join(rootpath, testCollection, 'ImageSets', '%s.txt'%testset)
    imset = map(str.strip, open(imsetfile).readlines())
    return imset