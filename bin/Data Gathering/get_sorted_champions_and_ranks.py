from os import walk
import json

f = []
for (dirpath, dirnames, filenames) in walk("../../matches/"):
    f.extend(filenames)
    break

def fillIn(array, cid, lane, role, rank):
    if lane == "TOP":
        array[0] = cid
        array[1] = rank
        return
    if lane == "JUNGLE":
        array[2] = cid
        array[3] = rank
        return
    if lane == "MIDDLE":
        array[4] = cid
        array[5] = rank
        return
    if lane == "BOTTOM":
        if role == "DUO_CARRY":
            array[6] = cid
            array[7] = rank
            return
        if role == "DUO_SUPPORT":
            array[8] = cid
            array[9] = rank
            return
    return -1

def allNonZero(array):
    for x in array:
        if x == 0:
            return False
    return True

newf = open("../../Data/champions+ranks.txt", "w")

count = 0
i = 0
for fileName in f:
    with open("../../matches/" + fileName) as f:
        contents = f.read()
    ds = json.loads(contents)

    blue = [0]*10
    red = [0]*10

    bluewin = int(ds['teams'][0]['win'] == "Win")

    for j in range(10):
        cid = ds['participants'][j]['championId']
        lane = ds['participants'][j]['timeline']['lane']
        role = ds['participants'][j]['timeline']['role']
        rank = ds['participants'][j]['highestAchievedSeasonTier']
        if j < 5:
            out = fillIn(blue, cid, lane, role, rank)
        else:
            out = fillIn(red, cid, lane, role, rank)
        #print((cid, lane, role))
    if allNonZero(blue) and allNonZero(red):
        #print >> newf, " ".join([str(x) for x in blue]) + "\t" + " ".join([str(x) for x in red]) + "\t" + str(bluewin)
        print(" ".join([str(x) for x in blue]) + "\t" + " ".join([str(x) for x in red]) + "\t" + str(bluewin), file=newf)
        count += 1
        
    i += 1

    if i % 1000 == 0:
        print(i)

newf.close()

print(count)
