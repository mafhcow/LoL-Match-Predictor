import torch
import random
import numpy as np
from mappings import rank_to_one_hot, id_to_one_hot
from Embedder import Embedder
'''
def parse_line(line):
	blue_champs, red_champs, result = line.split('\t')
	blue_champs = [int(x) for x in blue_champs.split(' ')]
	red_champs = [int(x) for x in red_champs.split(' ')]
	result = int(result)
	return blue_champs, red_champs, result

class DataBuilder:
	def __init__(self, data_path='../Data/champions.txt', train_test_split=0.75):
		self.split = train_test_split
		with open(data_path, 'r') as f:
			data = []
			for line in f.readlines():
				data.append(parse_line(line))
		random.shuffle(data)
		data_train = data[:int(self.split*len(data))]
		self.data = data
		self.embedder = Embedder(data_train)
		
	def build_data(self, embed=False):
		inputs = []
		outputs = []
		for blue_champs, red_champs, result in self.data:
			blue_vector = []
			red_vector = []
			for x in blue_champs:
				if embed:
					blue_vector = blue_vector + self.embedder.embed(x)
				else:
					blue_vector = blue_vector + id_to_one_hot(x)
			for x in red_champs:
				if embed:
					red_vector = red_vector + self.embedder.embed(x)
				else:
					red_vector = red_vector + id_to_one_hot(x)
			inputs.append(blue_vector + red_vector)
			outputs.append(result)
		
		l = len(inputs)
		input_train = inputs[:int(self.split*l)]
		input_test = inputs[int(self.split*l):]
		output_train = outputs[:int(self.split*l)]
		output_test = outputs[int(self.split*l):]
			
		return torch.FloatTensor(np.array(input_train)), torch.FloatTensor(np.array(input_test)), torch.LongTensor(np.array(output_train)), torch.LongTensor(np.array(output_test))
'''

def parse_line(line):
	blue_info, red_info, result = line.split('\t')
	blue_info = blue_info.split(' ')
	red_info = red_info.split(' ')
	blue_champs = []
	blue_ranks = []
	red_champs = []
	red_ranks = []
	for i in range(10):
		if i % 2 == 0:
			blue_champs.append(int(blue_info[i]))
			red_champs.append(int(red_info[i]))
		else:
			blue_ranks.append(rank_to_one_hot(blue_info[i]))
			red_ranks.append(rank_to_one_hot(red_info[i]))
	result = int(result)
	return blue_champs, blue_ranks, red_champs, red_ranks, result

class DataBuilder:
	def __init__(self, data_path='../Data/champions+ranks.txt', train_test_split=0.75):
		self.split = train_test_split
		with open(data_path, 'r') as f:
			data = []
			for line in f.readlines():
				data.append(parse_line(line))
		random.shuffle(data)
		no_rank_data = []
		for game in data:
			no_rank_data.append((game[0], game[2], game[4]))
		no_rank_data_train = no_rank_data[:int(self.split*len(data))]
		self.data = data
		self.embedder = Embedder(no_rank_data_train)
		
	def build_data(self, embed=False):
		inputs = []
		outputs = []
		for blue_champs, blue_ranks, red_champs, red_ranks, result in self.data:
			blue_vector = []
			red_vector = []
			for i in range(5):
				if embed:
					blue_vector = blue_vector + self.embedder.embed(blue_champs[i]) + blue_ranks[i]
					red_vector = red_vector + self.embedder.embed(red_champs[i]) + red_ranks[i]
				else:
					blue_vector = blue_vector + id_to_one_hot(blue_champs[i]) + blue_ranks[i]
					red_vector = red_vector + id_to_one_hot(red_champs[i]) + red_ranks[i]
			inputs.append(blue_vector + red_vector)
			outputs.append(result)
		
		l = len(inputs)
		input_train = inputs[:int(self.split*l)]
		input_test = inputs[int(self.split*l):]
		output_train = outputs[:int(self.split*l)]
		output_test = outputs[int(self.split*l):]
			
		return torch.FloatTensor(np.array(input_train)), torch.FloatTensor(np.array(input_test)), torch.LongTensor(np.array(output_train)), torch.LongTensor(np.array(output_test))

		
