import numpy as np
import json, os
import random
import gzip
import math
import struct
import pandas as pd
from array import array

class Tensorflow_data:
	def __init__(self, data_path, input_train_dir, set_name):
		#get product/user/vocabulary information
		self.product_ids = []
		with gzip.open(data_path + 'product.txt.gz', 'rt') as fin:
			for line in fin:
				self.product_ids.append(line.strip())
		self.product_size = len(self.product_ids)
		self.user_ids = []
		with gzip.open(data_path + 'users.txt.gz', 'rt') as fin:
			for line in fin:
				self.user_ids.append(line.strip())
		self.user_size = len(self.user_ids)
		self.words = []
		with gzip.open(data_path + 'vocab.txt.gz', 'rt') as fin:
			for line in fin:
				self.words.append(line.strip())
		self.vocab_size = len(self.words)

		#get review sets
		self.word_count = 0
		self.vocab_distribute = np.zeros(self.vocab_size) 
		self.review_info = []
		self.review_text = []
		with gzip.open(input_train_dir + set_name + '.txt.gz', 'rt') as fin:
			for line in fin:
				arr = line.strip().split('\t')
				self.review_info.append((int(arr[0]), int(arr[1]))) # (user_idx, product_idx)
				self.review_text.append([int(i) for i in arr[2].split(' ')])
				for idx in self.review_text[-1]:
					self.vocab_distribute[idx] += 1
				self.word_count += len(self.review_text[-1])
		self.review_size = len(self.review_info)
		self.vocab_distribute = self.vocab_distribute.tolist() 
		self.sub_sampling_rate = None
		self.review_distribute = np.ones(self.review_size).tolist()
		self.product_distribute = np.ones(self.product_size).tolist()

		print("Data statistic: vocab %d, review %d, user %d, product %d\n" % (self.vocab_size, 
					self.review_size, self.user_size, self.product_size))

	def sub_sampling(self, subsample_threshold):
		self.sub_sampling_rate = np.ones(self.vocab_size)
		if subsample_threshold == 0.0:
			return
		threshold = sum(self.vocab_distribute) * subsample_threshold
		count_sub_sample = 0
		for i in range(self.vocab_size):
			#vocab_distribute[i] could be zero if the word does not appear in the training set
			self.sub_sampling_rate[i] = min((np.sqrt(float(self.vocab_distribute[i]) / threshold) + 1) * threshold / float(self.vocab_distribute[i]),
											1.0)
			count_sub_sample += 1

	def read_image_features(self, data_path):
		self.img_feature_num = 4096
		self.img_features = [None for i in range(self.product_size)]
		with open(data_path + 'product_image_feature.b', 'rb') as fin:
			for i in range(self.product_size):
				float_array = array('f')
				float_array.fromfile(fin, 4096)
				self.img_features[i] = list(float_array)

	def read_latent_factor(self, data_path):
		user_latent_factor_file_name = data_path + 'user_factors.csv'
		item_latent_factor_file_name = data_path + 'item_factors.csv'
		user = pd.read_csv(user_latent_factor_file_name).iloc[:, 1:]
		item = pd.read_csv(item_latent_factor_file_name).iloc[:, 1:]
		self.user_factors = user.values[:self.user_size]
		self.product_factors = item.values[:self.product_size]
		self.rate_factor_num = len(self.user_factors[0]) # 200
		print('Rate factor size ' + str(self.rate_factor_num))
		#return user.values, item.values

	def read_train_product_ids(self, data_path):
		self.user_train_product_set_list = [set() for i in range(self.user_size)]
		self.train_review_size = 0
		with gzip.open(data_path + 'train.txt.gz', 'rt') as fin:
			for line in fin:
				self.train_review_size += 1
				arr = line.strip().split('\t')
				self.user_train_product_set_list[int(arr[0])].add(int(arr[1]))


	def compute_test_product_ranklist(self, u_idx, original_scores, sorted_product_idxs, rank_cutoff):
		product_rank_list = []
		product_rank_scores = []
		rank = 0
		for product_idx in sorted_product_idxs:
			if product_idx in self.user_train_product_set_list[u_idx] or math.isnan(original_scores[product_idx]):
				continue
			product_rank_list.append(product_idx)
			product_rank_scores.append(original_scores[product_idx])
			rank += 1
			if rank == rank_cutoff:
				break
		return product_rank_list, product_rank_scores


	def output_ranklist(self, user_ranklist_map, user_ranklist_score_map, output_path, similarity_func):
		with open(os.path.join(output_path, 'test.'+similarity_func+'.ranklist'), 'w') as rank_fout:
			for u_idx in user_ranklist_map:
				user_id = self.user_ids[u_idx]
				for i in range(len(user_ranklist_map[u_idx])):
					product_id = self.product_ids[user_ranklist_map[u_idx][i]]
					rank_fout.write(user_id + ' Q0 ' + product_id + ' ' + str(i+1)
							+ ' ' + str(user_ranklist_score_map[u_idx][i]) + ' MultiViewEmbedding\n')

	def output_embedding(self, embeddings, output_file_name):
		with open(output_file_name,'w') as emb_fout:
			length = len(embeddings)
			if length < 1:
				return
			dimensions = len(embeddings[0])
			emb_fout.write(str(length) + '\n')
			emb_fout.write(str(dimensions) + '\n')
			for i in range(length):
				for j in range(dimensions):
					emb_fout.write(str(embeddings[i][j]) + ' ')
				emb_fout.write('\n')
				 





