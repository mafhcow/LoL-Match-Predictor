import re
import urllib2
from urllib2 import HTTPError
from threading import Thread, Lock
import time
import json
import pickle
import os.path

API_KEY = "<HIDDEN>"
GET_MATCH = "https://na1.api.riotgames.com/lol/match/v3/matches/{}?api_key={}"
MATCH_HIST = "https://na1.api.riotgames.com/lol/match/v3/matchlists/by-account/{}/recent?api_key={}"

def getResponse(url, delay = 0):
    try:
        return urllib2.urlopen(url).read()
    except HTTPError as e:
        print("Ratelimited, retrying. " + str(e))
        if e.code == 429:
            time.sleep(30) # should sleep in a smarter fashion?
            return getResponse(url)


def getMatches(html):
    if html == None:
        return None
    datastore = json.loads(html)
    matchIds = []
    for i in range(len(datastore['matches'])): # maybe fewer than 20 matches
        matchIds.append(datastore['matches'][i]['gameId'])
    return matchIds

def getMatchData(html):
    if html == None:
        return None
    datastore = json.loads(html)
    if datastore['queueId'] != 420:
        return None # not SR ranked
    if not(datastore['gameVersion'].startswith('7.24')):
        return None # only take current patch
    
    # for now ignore the actual match just get people
    people = []
    for i in range(10):
        aid = datastore['participantIdentities'][i]['player']['accountId']
        people.append(aid)
    return people

if os.path.isfile("users.pickle"):
    with open("users.pickle", "rb") as f:
        users = pickle.load(f)
else:
    users = [29512] # pobelter

if os.path.isfile("matches.pickle"):
    with open("matches.pickle", "rb") as f:
        matches = pickle.load(f)
else:
    matches = []

if os.path.isfile("sDict.pickle"):
    with open("sDict.pickle", "rb") as f:
        sDict = pickle.load(f)
else:
    sDict = {29512}

if os.path.isfile("mDict.pickle"):
    with open("mDict.pickle", "rb") as f:
        mDict = pickle.load(f)
else:
    mDict = {0}

for i in range(1000000):
    if ((i + 1) % 10) == 0:
        with open("users.pickle", "wb") as f:
            pickle.dump(users, f)
        with open("matches.pickle", "wb") as f:
            pickle.dump(matches, f)
        with open("sDict.pickle", "wb") as f:
            pickle.dump(sDict, f)
        with open("mDict.pickle", "wb") as f:
            pickle.dump(mDict, f)
        print("----------------------------------------------------------")
        print("Currently on iteration: {}".format(i))

    #if len(users) != 0:
    if len(matches) == 0: # look for more matches
        if len(users) == 0:
            break
        u = users[-1]
        print(u)
        users.pop(-1)
        html = getResponse(MATCH_HIST.format(u, API_KEY))
        umatches = getMatches(html)
        if umatches == None:
            continue
        for match in umatches:
            if match in mDict:
                continue
            else:
                mDict.add(match)
                matches.append(match)
    
    m = matches[-1]
    matches.pop(-1)
    html = getResponse(GET_MATCH.format(m, API_KEY))
    
    people = getMatchData(html)
    if people == None:
        continue
    print(people)
    for p in people:
        if p in sDict:
            continue
        else:
            users.append(p)
            sDict.add(p)

    f = open("matches/{}.json".format(m), "w")
    f.write(html)
    f.close()
    
