import json
import os

def get_champions_from_file(filename):
        file = open('../matches/{0}'.format(filename), 'r', encoding='utf-8')
        text = file.read()
        parsed = json.loads(text)
        if len(parsed['participants']) != 10:
            print("Error: not 5 v 5 game (participants: {0})! ID: {1}".format(len(parsed['participants']), filename))
            return None, None, None
        if parsed['queueId'] != 420:
            print("Error: not ranked. ID: {0}".format(filename))
            return None, None, None
        blue_champions = [str(parsed['participants'][x]['championId']) for x in range(5)]
        red_champions = [str(parsed['participants'][x]['championId']) for x in range(5, 10)]
        result = parsed['teams'][0]['win']
        if result == 'Fail':
            result = 0
        else:
            result = 1
        return blue_champions, red_champions, result

def parse(blue_champions, red_champions, result):
    return ' '.join(blue_champions) + '\t' + ' '.join(red_champions) + '\t' + str(result)
    

def get_all_champions_from_files():
    results = []
    data_files = [x for x in os.listdir('../matches') if x.endswith('.json')]

    for filename in data_files:
        blue_champions, red_champions, result = get_champions_from_file(filename)
        if blue_champions is not None:
            results.append(parse(blue_champions, red_champions, result))

    return results

out_file = open('../Data/champions.txt', 'w')
results = get_all_champions_from_files()
for result in results:
    out_file.write(result + '\n')

out_file.close()
