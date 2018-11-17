import torch
import numpy as np
from PIL import Image, ImageDraw
PAD_TOKEN = -1
DIM = 2
MEAN = 300
VAR = 300
if torch.cuda.is_available():
	dtype = torch.cuda.FloatTensor
else:
	dtype = torch.FloatTensor
def generateGT(feature):
	#input:feature_size (batch_size, feature_dim)
	#output:V_g x dim_size
	#feature = feature#[feature != PAD_TOKEN]
	shape  = feature.shape
	feature = np.reshape(feature,(shape[0], int(shape[1]/DIM), DIM))
	return feature

def generateNormals(test=False):
	if test:
		file = 'normals_test.dat'
	else:
		file = 'normals_train.dat'
	max_length = 0
	with open(file) as f:
		lines=f.readlines()
		for line in lines:
			data_line = np.fromstring(line, dtype=float, sep=',')
			max_length = max(max_length,len(data_line))
		f.close()
	data = np.array([])
	max_length = 10*max_length
	with open(file) as f:
		lines=f.readlines()
		for line in lines:
			data_line = np.fromstring(line, dtype=float, sep=',')
			data_line = np.expand_dims(np.pad(data_line,(0,max_length-len(data_line)),'constant',constant_values=(0,10*PAD_TOKEN)),0)
			if(len(data)==0):
				data = data_line
			else:
				data = np.concatenate((data,data_line),axis=0)
		f.close()
	return data

def getData(test=False):
	if test:
		file = 'polygons_test.dat'
	else:
		file = 'polygons_train.dat'
	max_length = 0
	with open(file) as f:
		lines=f.readlines()
		for line in lines:
			data_line = np.fromstring(line, dtype=float, sep=',')
			max_length = max(max_length,len(data_line))
		f.close()
	data = np.array([])
	seq_len = np.array([])
	max_length = 10*max_length
	with open(file) as f:
		lines=f.readlines()
		for line in lines:
			data_line = np.fromstring(line, dtype=float, sep=',')
			cur_seq_len = int(len(data_line)/DIM)
			data_line = np.expand_dims(np.pad(data_line,(0,max_length-len(data_line)),'constant',constant_values=(0,PAD_TOKEN)),0)
			data_line[data_line==PAD_TOKEN] = PAD_TOKEN*VAR + MEAN
			data_line = (data_line - MEAN)/VAR
			if(len(data)==0):
				data = data_line
				seq_len = np.array([cur_seq_len])
			else:
				data = np.concatenate((data,data_line),axis=0)
				seq_len = np.concatenate((seq_len, np.array([cur_seq_len])),axis=0)
		f.close()

	data_normals = generateNormals(test=test)
	return data, data_normals, seq_len, max_length, DIM

def inputMesh(feature_size):
	c1= np.expand_dims(np.array([0,-0.9]),0)
	c2= np.expand_dims(np.array([-0.9,0.9]),0)
	c3= np.expand_dims(np.array([0.9,0.9]),0)
	f1 = np.expand_dims(np.pad(np.array([0,-0.9]),(0,feature_size-2),'constant',constant_values=(0,0)),0)
	f2 = np.expand_dims(np.pad(np.array([-0.9,0.9]),(0,feature_size-2),'constant',constant_values=(0,0)),0)
	f3 = np.expand_dims(np.pad(np.array([0.9,0.9]),(0,feature_size-2),'constant',constant_values=(0,0)),0)
	A = np.ones((3,3))
	A[0,0] = 0
	A[1,1] = 0
	A[2,2] = 0
	return np.concatenate((c1,c2,c3),axis=0), np.concatenate((f1,f2,f3),axis=0),A
def getPixels(c):
	return (c*VAR + MEAN).tolist()
def drawPolygons(polygons,polygonsgt,color='red',out='out.png',A=None):
	black = (0,0,0)
	white=(255,255,255)
	im = Image.new('RGB', (600, 600), white)
	imPxAccess = im.load()
	draw = ImageDraw.Draw(im,'RGBA')
	verts = polygons
	vertsgt = polygonsgt
	# either use .polygon(), if you want to fill the area with a solid colour
	points = tuple(tuple(x) for x in verts)
	#draw.point((points),fill=(255,0,0,0))
	for point in points:
	    draw.ellipse((point[0] - 4, point[1] - 4, point[0]  + 4, point[1] + 4), fill=color)
	if A is None:
		draw.polygon((points), outline=black,fill=(0,0,0,0) )
	else:
	# # or .line() if you want to control the line thickness, or use both methods together!
		for i in range(len(verts)):
			for j in range(len(verts)):
				#print(A)
				if(A[i,j]):
					
					draw.line((tuple(verts[i]),tuple(verts[j])), width=2, fill=black )
	color = 'green'					
	verts = vertsgt
	points = tuple(tuple(x) for x in verts)
	#draw.point((points),fill=(255,0,0,0))
	for point in points:
	    draw.ellipse((point[0] - 4, point[1] - 4, point[0]  + 4, point[1] + 4), fill=color)
	draw.polygon((points), outline='green',fill=(0,0,0,0) )
	im.save(out)
