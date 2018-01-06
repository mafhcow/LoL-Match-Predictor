import random

def power_mean(arr, r):
	ret = 0.0
	for element in arr:
		ret += element**r
	ret /= len(arr)
	return ret**(1.0/r)

def parse_line(line):
    blue_champs, red_champs, result = line.split('\t')
    blue_champs = [int(x) for x in blue_champs.split(' ')]
    red_champs = [int(x) for x in red_champs.split(' ')]
    result = int(result)
    return (blue_champs, red_champs, result)

def test_baseline(trials, r):
	total_accuracy = 0
	for i in range(trials):
		if i%(trials/10) == 0:
			print(str(int(100*i/trials)) + '% done')
		data = []
		with open('../Data/champions.txt', 'r') as f:
				for line in f.readlines():
					data.append(parse_line(line))
		random.shuffle(data)
		N = len(data)
		training_data = data[:int(3*N/4)]
		test_data = data[int(3*N/4):]

		champ_wins = {}
		champ_games = {}
		champ_winrates = {}
		for sample in training_data:
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
			
		accuracy = 0
		for sample in test_data:
			blue_score = 0
			red_score = 0
			blue_winrates = []
			red_winrates = []
			for i in range(5):
				blue_winrates.append(champ_winrates[sample[0][i]])
				red_winrates.append(champ_winrates[sample[1][i]])
			blue_score = power_mean(blue_winrates, r)
			red_score = power_mean(red_winrates, r)
			if (blue_score >= red_score and sample[2] == 1) or (red_score > blue_score and sample[2] == 0):
				accuracy += 1
		total_accuracy += 1.0*accuracy/len(test_data)
	return total_accuracy/trials

print('baseline accuracy is', test_baseline(100, -2))

