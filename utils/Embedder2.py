import json
import os
from mappings import id_to_num, num_to_id, id_to_name, all_ids, num_champs
import pickle
import numpy as np
from sklearn.decomposition import PCA

NUM_CHAMPS = num_champs()
champ_ids = all_ids()

class Embedder:
	def __init__(self, n_components=50, test_train_split=0.75):
		self.n_components = n_components
		self.split = test_train_split
		matches = os.listdir('../matches')
		champ_info = {}
		'''
		for id in champ_ids:
			champ_info[id] = {'n':0, 'winrate':0, 'physical_dmg':0, 'magic_dmg':0, 'true_dmg':0, 'kills':0, 'deaths':0, 'assists':0, 
								'wards_placed':0, 'cc_time':0, 'cs':0, 'jg_cs':0, 'counter_jg_cs':0, 'healing':0, 'dmg_taken':0}

		count = 0						
		for match in matches[:int(len(matches) * self.split)]:
			count += 1
			if count % 1000 == 0:
				print(count)
			with open('../matches/' + match, 'rb') as f:
				text = f.read().decode('utf8')
				parsed = json.loads(text)
				if len(parsed['participants']) == 10 and parsed['queueId'] == 420:
					game_time = 1.0*parsed['gameDuration']
					for i in range(10):
						id = parsed['participants'][i]['championId']
						win = parsed['participants'][i]['stats']['win']
						if win:
							win = 1.0
						else:
							win = 0.0
						physical_dmg = parsed['participants'][i]['stats']['physicalDamageDealtToChampions']/game_time
						magic_dmg = parsed['participants'][i]['stats']['magicDamageDealtToChampions']/game_time
						true_dmg = parsed['participants'][i]['stats']['trueDamageDealtToChampions']/game_time
						kills = parsed['participants'][i]['stats']['kills']/game_time
						deaths = parsed['participants'][i]['stats']['deaths']/game_time
						assists = parsed['participants'][i]['stats']['assists']/game_time
						wards_placed = parsed['participants'][i]['stats']['wardsPlaced']/game_time
						cc_time = parsed['participants'][i]['stats']['timeCCingOthers']/game_time
						cs = parsed['participants'][i]['stats']['totalMinionsKilled']/game_time
						jg_cs = parsed['participants'][i]['stats']['neutralMinionsKilledTeamJungle']/game_time
						counter_jg_cs = parsed['participants'][i]['stats']['neutralMinionsKilledEnemyJungle']/game_time
						healing = parsed['participants'][i]['stats']['totalUnitsHealed']/game_time
						dmg_taken = parsed['participants'][i]['stats']['totalDamageTaken']/game_time
						
						champ_info[id]['winrate'] = (champ_info[id]['winrate']*champ_info[id]['n'] + win)/(champ_info[id]['n'] + 1)
						champ_info[id]['physical_dmg'] = (champ_info[id]['physical_dmg']*champ_info[id]['n'] + physical_dmg)/(champ_info[id]['n'] + 1)
						champ_info[id]['magic_dmg'] = (champ_info[id]['magic_dmg']*champ_info[id]['n'] + magic_dmg)/(champ_info[id]['n'] + 1)
						champ_info[id]['true_dmg'] = (champ_info[id]['true_dmg']*champ_info[id]['n'] + true_dmg)/(champ_info[id]['n'] + 1)
						champ_info[id]['kills'] = (champ_info[id]['kills']*champ_info[id]['n'] + kills)/(champ_info[id]['n'] + 1)
						champ_info[id]['deaths'] = (champ_info[id]['deaths']*champ_info[id]['n'] + deaths)/(champ_info[id]['n'] + 1)
						champ_info[id]['assists'] = (champ_info[id]['assists']*champ_info[id]['n'] + assists)/(champ_info[id]['n'] + 1)
						champ_info[id]['wards_placed'] = (champ_info[id]['wards_placed']*champ_info[id]['n'] + wards_placed)/(champ_info[id]['n'] + 1)
						champ_info[id]['cc_time'] = (champ_info[id]['cc_time']*champ_info[id]['n'] + cc_time)/(champ_info[id]['n'] + 1)
						champ_info[id]['cs'] = (champ_info[id]['cs']*champ_info[id]['n'] + cs)/(champ_info[id]['n'] + 1)
						champ_info[id]['jg_cs'] = (champ_info[id]['jg_cs']*champ_info[id]['n'] + jg_cs)/(champ_info[id]['n'] + 1)
						champ_info[id]['counter_jg_cs'] = (champ_info[id]['counter_jg_cs']*champ_info[id]['n'] + counter_jg_cs)/(champ_info[id]['n'] + 1)
						champ_info[id]['healing'] = (champ_info[id]['healing']*champ_info[id]['n'] + healing)/(champ_info[id]['n'] + 1)
						champ_info[id]['dmg_taken'] = (champ_info[id]['dmg_taken']*champ_info[id]['n'] + dmg_taken)/(champ_info[id]['n'] + 1)
						
						champ_info[id]['n'] = champ_info[id]['n'] + 1

		with open('../Data/champ_train_info.pkl', 'wb') as f:
			pickle.dump(champ_info, f)'''

		with open('../Data/champ_train_info.pkl', 'rb') as f:
			champ_info = pickle.load(f)

		champ_info_matrix = []
		for i in range(NUM_CHAMPS):
			champ_info_matrix.append([])
		for id in champ_ids:
			num = id_to_num(id)
			for key in champ_info[id]:
				if key != 'n':
					champ_info_matrix[num].append(champ_info[id][key])
					
		champ_info_matrix = np.array(champ_info_matrix)
		champ_info_matrix = (champ_info_matrix - champ_info_matrix.mean(axis=0))/champ_info_matrix.std(axis=0)

		def sim(v1, v2):
			return np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2))

		sim_matrix = []
		for i in range(NUM_CHAMPS):
			sim_matrix.append([0]*NUM_CHAMPS)
		for i in range(NUM_CHAMPS):
			for j in range(NUM_CHAMPS):
				sim_matrix[i][j] = sim(champ_info_matrix[i], champ_info_matrix[j])

		'''c = 34
		for i in range(NUM_CHAMPS):
			most_sim = sorted(champ_ids, key = lambda x: sim(champ_info_matrix[id_to_num(x)], champ_info_matrix[id_to_num(c)]), reverse=True)
		for el in most_sim:
			print(id_to_name(el), sim(champ_info_matrix[id_to_num(el)], champ_info_matrix[id_to_num(c)]))'''
			
		pca = PCA(n_components = n_components)
		pca.fit(sim_matrix)
		sim_matrix = pca.transform(sim_matrix)
		self.sim_matrix = sim_matrix
	
	def embed(self, id):
		return list(self.sim_matrix[id_to_num(id)])




	

				
				
				
			
		
		
