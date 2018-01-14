from os import walk
import json

f = []
for (dirpath, dirnames, filenames) in walk("../../matches/"):
    f.extend(filenames)
    break

def fillIn(array, cid, lane, role):
    if lane == "TOP":
        array[0] = cid
        return
    if lane == "JUNGLE":
        array[1] = cid
        return
    if lane == "MIDDLE":
        array[2] = cid
        return
    if lane == "BOTTOM":
        if role == "DUO_CARRY":
            array[3] = cid
            return
        if role == "DUO_SUPPORT":
            array[4] = cid
            return
    return -1

def allNonZero(array):
    for x in array:
        if x == 0:
            return False
    return True

newf = open("../../Data/champions.txt", "w")

count = 0
i = 0
for fileName in f:
    with open("../../matches/" + fileName) as f:
        contents = f.read()
    ds = json.loads(contents)

    blue = [0,0,0,0,0]
    red = [0,0,0,0,0]

    bluewin = int(ds['teams'][0]['win'] == "Win")

    for j in range(10):
        cid = ds['participants'][j]['championId']
        lane = ds['participants'][j]['timeline']['lane']
        role = ds['participants'][j]['timeline']['role']
        if j < 5:
            out = fillIn(blue, cid, lane, role)
        else:
            out = fillIn(red, cid, lane, role)
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
