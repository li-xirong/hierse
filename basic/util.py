
import os
from numpy import zeros
from common import readRankingResults,ROOT_PATH,printStatus,printError

def readWordnetVob(collection, rootpath=ROOT_PATH):
    vobfile = os.path.join(rootpath, collection, "TextData", "wn.%s.txt" % collection)
    return map(str.strip, open(vobfile).readlines())
    

def getsubset(sourcefiles, imset, resultfile):
    print ("request %d" % len(imset))
    cached = set()

    try:
        os.makedirs(os.path.split(resultfile)[0])
    except:
        pass
       
    fout = open(resultfile, 'w')
    output = []

    for sourcefile in sourcefiles:
        print ("parsing " + sourcefile)
        for line in open(sourcefile): 
            elems = str.split(line.strip())
            imageid = elems[0]
            if (imageid in cached) or (imset and (imageid not in imset)):
                continue
            cached.add(imageid)
            output.append(line)
        if len(output) % 5e4 == 0:
            print (len(cached))
            fout.write("".join(output))
            output = []
    if output:
        fout.write("".join(output))
        output = []
    fout.close()

    print ('%d requested, %d obtained' % (len(imset), len(cached)))

def getsubsetf(sourcefiles, imsetfile, resultfile):
    imset = set([str.split(x)[0] for x in open(imsetfile).readlines()])
    getsubset(sourcefiles, imset, resultfile)


'''
scoreTable =
          image_1 image_2 ... image_n
feature_1
feature_2
.
.
.
feature_m
'''
def readImageScoreTable(concept, name2index, similarityIndexDir, models, torank):
    numModels = len(models)
    numInstances = len(name2index)
    scoreTable = zeros((numModels, numInstances))

    for i in range(numModels):
        [modelName, weight, scale] = models[i]

        scorefile = os.path.join(similarityIndexDir, modelName, concept + ".txt")
        searchResults = readRankingResults(scorefile)
        searchResults = [x for x in searchResults if x[0] in name2index]
        for rank, (name,score) in enumerate(searchResults):
            if torank:
                score = 1.0 - float(rank)/numInstances
            else:
                score /= scale
            scoreTable[i, name2index[name]] = score

    return scoreTable       
    

'''
    n: number of nodes in a valid path
	m: target value, wi=0,1,...,m, for i=1,...,n
	Si: the summation at node_i, i=0,...,n, where S0=0, Si<=m, and Sn=m.
'''
def searchpath(n, m):
    paths = [(0, [])]
	
    for i in range(1, n+1):
        #print i, i-1, paths
        newpaths = []
        for s,path in paths:
            if i == n:
                path.append(m-s)
                newpaths.append((m,path))
            else:    
                if s == m:
                    path.append(0)
                    newpaths.append((s,path))
                else:
                    for w in range(m-s+1):
                        newpath = list(path)
                        newpath.append(w)    
                        newpaths.append((s+w,newpath))
        paths = newpaths
        #print i, paths

    return paths
        
def readImageSet(collection, dataset=None, rootpath=ROOT_PATH):
    if not dataset:
        dataset = collection
    imsetfile = os.path.join(rootpath, collection, 'ImageSets', '%s.txt' % dataset)
    imset = map(str.strip, open(imsetfile).readlines())
    return imset

def readLabeledImageSet(collection, tag, tpp='lemm', rootpath=ROOT_PATH):
    datafile = os.path.join(rootpath, collection, 'tagged,%s'% tpp, '%s.txt' % tag)
    try:
        hitset = map(str.strip, open(datafile).readlines())
        printStatus('basic.util.readLabeledImageSet', '%s-%s -> %d hits' % (collection, tag, len(hitset)))
    except:
        printError('basic.util.readLabeledImageSet', 'failed to read %s' % datafile)
        hitset = []
    return hitset


if __name__ == '__main__':
   collection = 'flickr81'
   imset = readImageSet(collection)
   print collection, len(imset)
   readLabeledImageSet('tagreldemo', 'airplane')
   
