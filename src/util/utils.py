import numpy as np
from PIL import Image, ImageDraw
MEAN = 300
VAR = 300
PAD_TOKEN = -2
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

def drawPolygons(polygons,polygonsgt, proj_pred=None, proj_gt=None, color='red',out='out.png',A=None):
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
		    draw.ellipse((point[0] - 4, point[1] - 4, point[0]  + 4, point[1] + 4), fill='green')
		draw.polygon((points), outline='green',fill=(0,0,0,0))

	# Shadow
	if proj_gt is not None:
		for i in range(im.size[0]):
			for j in range(im.size[1]-10,im.size[1]):
				imPxAccess[i,j] = (0,int(proj_gt[i])*128,0)
	if proj_pred is not None:
		for i in range(im.size[0]):
			for j in range(im.size[1]-20,im.size[1]-10):
				imPxAccess[i,j] = (int(proj_pred[i])*255,0,0)
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

def project_1d(polygons_data_line,params):
	proj_data_line = np.zeros(params.img_width,dtype=float)
	feature_size = len(polygons_data_line[0])
	p = 0
	minx = params.img_width -1 
	maxx = 0
	while True:
		if p < feature_size-2 and polygons_data_line[0,p] == PAD_TOKEN and polygons_data_line[0,p+2] == PAD_TOKEN:
			proj_data_line[minx:maxx+1] = 1.0
			break
		if p >= feature_size:
			proj_data_line[minx:maxx+1] = 1.0
			break
		if polygons_data_line[0,p] == PAD_TOKEN:
			p += 2
			proj_data_line[minx:maxx+1] = 1.0
			minx = params.img_width -1
			maxx = 0
			continue
		minx = min(minx,int(polygons_data_line[0,p]))
		maxx = max(maxx,int(polygons_data_line[0,p]))
		p += 2
	proj_data_line = np.expand_dims(proj_data_line,axis = 0)
	return proj_data_line

def flatten_pred(c,A,params):
	#c:  num_vertsx2
	#A:  num_vertsxnum_verts
	num_verts = c.shape[0]
	vertFlags = np.zeros(num_verts)
	proj_data_line = np.zeros(params.img_width,dtype=float)
	v = 0
	minx = params.img_width - 1
	maxx = 0
	
	start = v
	minx = min(minx,c[v,0])
	maxx = max(maxx,c[v,0])
	vertFlags[v] = 1
	for j in range(num_verts):
		if A[v,j] and vertFlags[j]==0:
			v = j
			break
	while True:
		minx = min(minx,c[v,0])
		maxx = max(maxx,c[v,0])
		vertFlags[v] = 1
		found_nbr = False
		for j in range(num_verts):
			if A[v,j] and vertFlags[j]==0:
				v = j
				found_nbr = True
				break
		if not found_nbr:
			proj_data_line[minx:maxx+1] = 1.0
			minx = params.img_width - 1
			maxx = 0
			found_poly = False
			for j in range(num_verts):
				if vertFlags[j]==0:
					v = j
					found_poly = True
					break
			if not found_poly:
				break
	return proj_data_line


def flatten_pred_batch(c,A,params):
	proj_batch = None
	for i in range(len(c)):
		proj = flatten_pred(np.array(c[i],dtype=int), A[i], params)
		proj = np.expand_dims(proj, axis=0)
		if proj_batch is None:
			proj_batch = proj
		else:
			proj_batch = np.concatenate((proj_batch, proj), axis=0)
	return proj_batch