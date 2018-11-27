import numpy as np
from PIL import Image, ImageDraw
MEAN = 300
VAR = 300
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
	i = 0
	polygons=[[]]
	for point in points:
		if point[0] >= 0 and point[1] >= 0:
			polygons[-1].append(point)
		else:
			polygons.append([])
	#draw.point((points),fill=(255,0,0,0))
	for points in polygons:
		for point in points:
		    draw.ellipse((point[0] - 4, point[1] - 4, point[0]  + 4, point[1] + 4), fill=color)
		draw.polygon((points), outline='green',fill=(0,0,0,0) )
	im.save(out)


def reshapeGT(params, feature):
	#input:feature_size (batch_size, feature_dim)
	#output:V_g x dim_size
	#feature = feature#[feature != PAD_TOKEN]
	shape  = feature.shape
	feature = np.reshape(feature,(shape[0], int(shape[1]/params.dim_size), params.dim_size))
	return feature
	
def create_mask(gt, seq_len):
	# seq_len: batch_size x 1
	batch_size = seq_len.shape[0]
	mask = np.arange(gt.size(1))
	mask = np.expand_dims(mask, 0)
	mask = np.tile(mask,(batch_size, 1))
	seq_len_check = np.reshape(seq_len, (-1, 1))
	seq_len_check = np.tile(seq_len_check,(1, gt.size(1)))
	condition = (mask < seq_len_check)
	return condition.astype(np.uint8)