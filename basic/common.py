import sys
import os
import time

#ROOT_PATH = "C:/Users/xirong/VisualSearch"
ROOT_PATH = "/home/lgp105b/xirong/VisualSearch"
ROOT_PATH = "/home/wdp/xirong/VisualSearch"
ROOT_PATH = "/Users/xirong/VisualSearch"


def makedirsforfile(filename):
    try:
        os.makedirs(os.path.split(filename)[0])
    except:
        pass


def niceNumber(v, maxdigit=6):
    """Nicely format a number, with a maximum of 6 digits."""
    assert(maxdigit >= 0)

    if maxdigit == 0:
        return "%.0f" % v

    fmt = '%%.%df' % maxdigit
    s = fmt % v
    
    if len(s) > maxdigit:
        return s.rstrip("0").rstrip(".")
    elif len(s) == 0:
        return "0"
    else:
        return s

def readRankingResults(filename):
    lines = open(filename).readlines()

    rankedList = []
    for line in lines:
        [imageid, score] = str.split(line.strip())[:2]
        rankedList.append((imageid, float(score)))
    return rankedList


def writeRankingResults(rankedList, filename):
    try:
        os.makedirs(os.path.split(filename)[0])
    except:
        pass
    fout = open(filename, "w")
    fout.write(''.join(['%s %s\n' % (imageid, niceNumber(score,8)) for (imageid, score) in rankedList]))
    fout.close()
        

def checkToSkip(filename, overwrite):
    if os.path.exists(filename):
        print ("%s exists." % filename),
        if overwrite:
            print ("overwrite")
            return 0
        else:
            print ("skip")
            return 1
    return 0    
    
    
def printMessage(message_type, trace, message):
    print ('%s %s [%s] %s' % (time.strftime('%d/%m/%Y %H:%M:%S'), message_type, trace, message))

def printStatus(trace, message):
    printMessage('INFO', trace, message)

def printError(trace, message):
    printMessage('ERROR', trace, message)


def total_seconds(td):
    return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 1e6) / 1e6

if __name__ == "__main__":
    print niceNumber(1.0/3, 4)
    for i in range(0, 15):
        print niceNumber(8.17717824342e-10, i)
        
        
