import torch
import numpy as np
import os

PAD_TOKEN = -1
MEAN = 300
VAR = 300
if torch.cuda.is_available():
	dtype = torch.cuda.FloatTensor
else:
	dtype = torch.FloatTensor

def getMetaData(params):
	f_meta = os.path.join(params.data_dir,'meta_%s.dat'%params.suffix)
	max_vertices = 0
	with open(f_meta) as f:
		line = f.readline().strip()
		max_vertices = int(line)
		line = f.readline().strip()
		data_size = int(line)
	return max_vertices, max_vertices*params.feature_scale*params.dim_size, data_size

def getDataLoader(params):
	f_polygons_path = os.path.join(params.data_dir,'polygons_%s.dat'%params.suffix)
	f_normals_path = os.path.join(params.data_dir,'normals_%s.dat'%params.suffix)
	_, feature_size, _ = getMetaData(params)
	iter_count = 0
	while True:
		f_polygons = open(f_polygons_path, 'r')
		f_normals = open(f_normals_path, 'r')

		polygons_data = np.array([])
		normals_data = np.array([])
		seq_len = np.array([])
		polygons_line = f_polygons.readline()
		normals_line = f_normals.readline()
		
		while(polygons_line != '' or normals_line != ''):
			iter_count += 1

			#polygons
			polygons_data_line = np.fromstring(polygons_line, dtype=float, sep=',')
			cur_seq_len = int(len(polygons_data_line)/params.dim_size)
			polygons_data_line = np.expand_dims(np.pad(polygons_data_line,(0,feature_size-len(polygons_data_line)),'constant',constant_values=(0,PAD_TOKEN)),0)
			polygons_data_line[polygons_data_line==PAD_TOKEN] = PAD_TOKEN*VAR + MEAN
			polygons_data_line = (polygons_data_line - MEAN)/VAR

			#normals
			normals_data_line = np.fromstring(normals_line, dtype=float, sep=',')
			normals_data_line = np.expand_dims(np.pad(normals_data_line,(0,feature_size-len(normals_data_line)),'constant',constant_values=(0,PAD_TOKEN)),0)
			
			if(len(polygons_data)==0):
				polygons_data = polygons_data_line
				normals_data = normals_data_line
				seq_len = np.array([cur_seq_len])
			else:
				polygons_data = np.concatenate((polygons_data, polygons_data_line),axis=0)
				normals_data = np.concatenate((normals_data, normals_data_line),axis=0)
				seq_len = np.concatenate((seq_len, np.array([cur_seq_len])),axis=0)
			if iter_count >= params.batch_size:
				yield polygons_data, normals_data, seq_len
				iter_count = 0
				polygons_data = np.array([])
				normals_data = np.array([])
				seq_len = np.array([])

			polygons_line = f_polygons.readline()
			normals_line = f_normals.readline()
		
		f_polygons_path.close()
		f_normals_path.close()