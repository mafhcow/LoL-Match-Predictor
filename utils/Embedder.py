import json
import numpy as np
from sklearn.decomposition import PCA
from mappings import num_to_id, id_to_num, num_champs

NUM_CHAMPS = num_champs()

class Embedder:
	def __init__(self, data, n_components=50):
		self.n_components = n_components
		champ_wins = {}
		champ_games = {}
		champ_winrates = {}
		for sample in data:
			for i in range(5):
				if sample[0][i] not in champ_wins:
					champ_wins[sample[0][i]] = sample[2]
				else:
					champ_wins[sample[0][i]] += sample[2]
				if sample[1][i] not in champ_wins:
					champ_wins[sample[1][i]] = 1 - sample[2]
				else:
					champ_wins[sample[1][i]] += (1 - sample[2])
				if sample[0][i] not in champ_games:
					champ_games[sample[0][i]] = 1
				else:
					champ_games[sample[0][i]] += 1
				if sample[1][i] not in champ_games:
					champ_games[sample[1][i]] = 1
				else:
					champ_games[sample[1][i]] += 1

		for champ in champ_games:
			champ_winrates[champ] = 1.0*champ_wins[champ]/champ_games[champ]

		champ_vs_wins = {}
		champ_vs_games = {}
		champ_vs_winrates = {}
		for sample in data:
			for i in range(5):
				for j in range(5):
					if (sample[0][i], sample[1][j]) in champ_vs_games:
						champ_vs_games[(sample[0][i], sample[1][j])] += 1
					else:
						champ_vs_games[(sample[0][i], sample[1][j])] = 1
					if sample[2] == 1:
						if (sample[0][i], sample[1][j]) not in champ_vs_wins:
							champ_vs_wins[(sample[0][i], sample[1][j])] = 1
						else:
							champ_vs_wins[(sample[0][i], sample[1][j])] += 1
					else:
						if (sample[1][i], sample[0][j]) not in champ_vs_wins:
							champ_vs_wins[(sample[1][i], sample[0][j])] = 1
						else:
							champ_vs_wins[(sample[1][i], sample[0][j])] += 1

		for (champ_1, champ_2) in champ_vs_games:
			if champ_vs_games[(champ_1, champ_2)] + champ_vs_games.get((champ_2, champ_1), 0) < 100:
				champ_vs_winrates[(champ_1, champ_2)] = champ_winrates[champ_1] - champ_winrates[champ_2] + 0.5
			else:
				champ_vs_winrates[(champ_1, champ_2)] = 1.0*champ_vs_wins.get((champ_1, champ_2), 0)/(champ_vs_games[(champ_1, champ_2)] + champ_vs_games.get((champ_2, champ_1), 0))

		embedding = {}
		for x in range(NUM_CHAMPS):
			embedding[x] = [0]*NUM_CHAMPS
			embedding[x][x] = 0.5
			for y in range(NUM_CHAMPS):
				if y != x:
					if (x, y) not in champ_vs_winrates:
						embedding[x][y] = 0
					else:
						embedding[x][y] = champ_vs_winrates[(x, y)] - champ_winrates[x] + champ_winrates[y] - 0.5


		def sim(e1, e2):
			arr1 = []
			arr2 = []
			for i in range(NUM_CHAMPS):
				if e1[i] != 0 and e2[i] != 0:
					arr1.append(e1[i])
					arr2.append(e2[i])
			a = 0
			b = 0
			for e in arr1:
				a += e**2
			for e in arr2:
				b += e**2
			c = 0
			for i in range(len(arr1)):
				c += arr1[i]*arr2[i]
			if a * b == 0:
				return 0
			return 1.0*c/(a * b)**0.5

		sim_matrix = []
		for i in range(NUM_CHAMPS):
			sim_matrix.append([0]*NUM_CHAMPS)
		for i in range(NUM_CHAMPS):
			for j in range(NUM_CHAMPS):
				sim_matrix[i][j] = sim(embedding[i], embedding[j])

		pca = PCA(n_components = self.n_components)
		pca.fit(sim_matrix)
		sim_matrix = pca.transform(sim_matrix)
		self.sim_matrix = sim_matrix
		
	def embed(self, id):
		return list(self.sim_matrix[id_to_num(id)])
	



			
			
				
			
				
	