import math, random
from PIL import Image, ImageDraw
import numpy as np
import argparse
import os
#from src.util import utils

def distance(a,b):
	return np.math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def parseArgs():
	parser = argparse.ArgumentParser(description='polygonGenerate.py')
	
	# General system running and configuration options  
	parser.add_argument('-p','--pad_token', type=str, default=',-2,-2,', help='Pad token to separate polygons in same data instance')
	parser.add_argument('-s','--suffix', type=str, default='train', help='suffix_name')
	parser.add_argument('-d','--data_size', type=int, default=1000, help='Data size')
	parser.add_argument('-n','--num_polygons', type=int, default=1, help='num of polygons per instance')
	parser.add_argument('-sig','--sigma', type=int, default=10, help='sigma')
	parser.add_argument('-r','--gen_random_polygons', dest='random_num_polygons', default=False, action='store_true', help='generate random number of polygons')
	parser.add_argument('-o','--no_overlap', dest='no_overlap', default=False, action='store_true', help='creates polygons with no overlap in y dimension')
	args = parser.parse_args()
	return args
def interpolate(a,b,length):
	dist = distance(a,b)
	n = int(dist*10/length)
	if n == 0:
		return []
	l = (np.array(list(b)) - np.array(list(a)))/n
	points = []
	curr = np.array(list(a))
	for i in range(n-1):
		points.append(tuple((curr+l).tolist()))
		curr = curr + l
	return points



def generatePolygon( ctrX, ctrY, aveRadius):
	'''Start with the centre of the polygon at ctrX, ctrY, 
		then creates the polygon by sampling points on a circle around the centre. 
		Randon noise is added by varying the angular spacing between sequential points,
		and by varying the radial distance of each point from the centre.

		Params:
		ctrX, ctrY - coordinates of the "centre" of the polygon
		aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
		irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
		spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
		numVerts - self-explanatory
		no_overlap - makes sure polygons don't overlap in y dimension of 2d image (can see separately in 1d projection)
		Returns a list of vertices, in CCW order.
	'''
	length = 1.8*aveRadius

	bot_len = 0.4*length + 0.05*length*np.random.randn()
	wing_offset1 = 0.05*length*np.random.randn()
	wing_offset2 = 0.1*length + 0.02*length*np.random.randn()
	wing_len = 0.1*length + abs(0.02*length*np.random.randn())
	wing_wid = 0.4*length + 0.05*length*np.random.randn()
	tail_len = 0.1*length + abs(0.02*length*np.random.randn())
	tail_wid = 0.2*length + 0.02*length*np.random.randn()
	thickness = 0.05*length + abs(0.02*length*np.random.randn())

	points = []
	y = ctrY
	points.append( (int(ctrX+tail_wid),int(y-length/2)) )
	points += interpolate((int(ctrX+tail_wid),int(y-length/2)),(int(ctrX+tail_wid),int(y-length/2+tail_len)),length)
	points.append( (int(ctrX+tail_wid),int(y-length/2+tail_len)) )
	points += interpolate((int(ctrX+tail_wid),int(y-length/2+tail_len)),(int(ctrX+thickness),int(y-length/2+tail_len)),length)
	points.append( (int(ctrX+thickness),int(y-length/2+tail_len)) )
	points += interpolate((int(ctrX+thickness),int(y-length/2+tail_len)),(int(ctrX+thickness),int(y-length/2+tail_len+bot_len)),length)
	points.append( (int(ctrX+thickness),int(y-length/2+tail_len+bot_len)) )
	points += interpolate((int(ctrX+thickness),int(y-length/2+tail_len+bot_len)),(int(ctrX+thickness+wing_wid),int(y-length/2+tail_len+bot_len+wing_offset1)),length)
	points.append( (int(ctrX+thickness+wing_wid),int(y-length/2+tail_len+bot_len+wing_offset1)) )
	points += interpolate((int(ctrX+thickness+wing_wid),int(y-length/2+tail_len+bot_len+wing_offset1)),(int(ctrX+thickness+wing_wid),int(y-length/2+tail_len+bot_len+wing_offset1+wing_len)),length)
	points.append( (int(ctrX+thickness+wing_wid),int(y-length/2+tail_len+bot_len+wing_offset1+wing_len)) )
	points += interpolate((int(ctrX+thickness+wing_wid),int(y-length/2+tail_len+bot_len+wing_offset1+wing_len)),(int(ctrX+thickness),int(y-length/2+tail_len+bot_len+wing_offset1+wing_len+wing_offset2)),length)
	points.append( (int(ctrX+thickness),int(y-length/2+tail_len+bot_len+wing_offset1+wing_len+wing_offset2)) )
	points += interpolate((int(ctrX+thickness),int(y-length/2+tail_len+bot_len+wing_offset1+wing_len+wing_offset2)),(int(ctrX),int(y+length/2)),length)
	points.append( (int(ctrX),int(y+length/2)) )
	points += interpolate((int(ctrX),int(y+length/2)),(int(ctrX-thickness),int(y-length/2+tail_len+bot_len+wing_offset1+wing_len+wing_offset2)), length)
	points.append( (int(ctrX-thickness),int(y-length/2+tail_len+bot_len+wing_offset1+wing_len+wing_offset2)) )
	points += interpolate((int(ctrX-thickness),int(y-length/2+tail_len+bot_len+wing_offset1+wing_len+wing_offset2)), (int(ctrX-thickness-wing_wid),int(y-length/2+tail_len+bot_len+wing_offset1+wing_len)), length)
	points.append( (int(ctrX-thickness-wing_wid),int(y-length/2+tail_len+bot_len+wing_offset1+wing_len)) )
	points += interpolate((int(ctrX-thickness-wing_wid),int(y-length/2+tail_len+bot_len+wing_offset1+wing_len)) ,(int(ctrX-thickness-wing_wid),int(y-length/2+tail_len+bot_len+wing_offset1)), length)
	points.append( (int(ctrX-thickness-wing_wid),int(y-length/2+tail_len+bot_len+wing_offset1)) )
	points+= interpolate((int(ctrX-thickness-wing_wid),int(y-length/2+tail_len+bot_len+wing_offset1)),(int(ctrX-thickness),int(y-length/2+tail_len+bot_len)), length)
	points.append( (int(ctrX-thickness),int(y-length/2+tail_len+bot_len)) )
	points += interpolate((int(ctrX-thickness),int(y-length/2+tail_len+bot_len)),(int(ctrX-thickness),int(y-length/2+tail_len)), length)
	points.append( (int(ctrX-thickness),int(y-length/2+tail_len)) )
	points += interpolate((int(ctrX-thickness),int(y-length/2+tail_len)),(int(ctrX-tail_wid),int(y-length/2+tail_len)), length)
	points.append( (int(ctrX-tail_wid),int(y-length/2+tail_len)) )
	points += interpolate((int(ctrX-tail_wid),int(y-length/2+tail_len)),(int(ctrX-tail_wid),int(y-length/2)), length)
	points.append( (int(ctrX-tail_wid),int(y-length/2)) )
	points += interpolate((int(ctrX-tail_wid),int(y-length/2)),(int(ctrX+tail_wid),int(y-length/2)), length)

	# generate n angle steps
	poly = np.array([list(point) for point in points])
	poly = poly -np.array([ctrX,ctrY])
	theta = np.random.uniform()*2*np.pi
	R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
	poly = np.matmul(poly,R)
	poly = poly +np.array([ctrX,ctrY])
	points = [tuple(point) for point in poly.tolist()]
	return points

def clip(x, min, max):
	if( min > max ) :  return x	
	elif( x < min ) :  return min
	elif( x > max ) :  return max
	else :			 return x

def distance(a,b):
	return np.math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def writePolygons(file,polygons, pad_token):
	"""
	Pads token to separate polygons in same data instance
	"""
	for p in range(len(polygons)):
		polygon  = polygons[p]
		for v in range(len(polygon)):
			vert = polygon[v]
			if(v == 0):
				file.write(str(vert[0])+','+str(vert[1]))
			else:
				file.write(','+str(vert[0])+','+str(vert[1]))
		if(p!=len(polygons)-1):
			file.write(pad_token)
	file.write('\n')

def normalise(v):
	normal2 = v
	return [normal2[0]/math.sqrt(normal2[1]**2+normal2[0]**2), normal2[1]/math.sqrt(normal2[1]**2+normal2[0]**2)]

def writeNormals(file, polygons, pad_token):
	"""
	Pads token to separate polygons in same data instance
	"""
	allnormals = []
	for p in range(len(polygons)):
		polygon  = polygons[p]
		polygonnormals = []
		for v in range(len(polygon)):
			vertp = polygon[v-1]
			vert = polygon[v]
			vertn = polygon[(v+1)%len(polygon)]
			normal1 = [vert[1]-vertp[1],vertp[0]-vert[0]]
			normal1 = normalise(normal1)

			normal2 = [vertn[1]-vert[1],vert[0]-vertn[0]]
			normal2 = normalise(normal2)
			normal = [0,0]
			normal[0] = normal1[0] + normal2[0]
			normal[1] = normal1[1] + normal2[1]
			normal = normalise(normal)
			if(v == 0):
				file.write(str(normal[0])+','+str(normal[1]))
			else:
				file.write(','+str(normal[0])+','+str(normal[1]))
			polygonnormals.append(normal)
		allnormals.append(polygonnormals)
		if(p!=len(polygons)-1):
			file.write(pad_token)			
	file.write('\n')
	return allnormals


def dataGenerator(params):
	data_size, suffix, total_polygons, pad_token = params.data_size, params.suffix, params.num_polygons, params.pad_token
	filepath  = "../../data/1" if total_polygons==1 else "../../data/2"
	f = open(os.path.join(filepath,'polygons_%s.dat'%suffix),'w')
	f_normal = open(os.path.join(filepath,'normals_%s.dat'%suffix),'w')
	num_polygons = total_polygons
	max_verts = 0

	for i in range(data_size):
		if params.random_num_polygons:
			num_polygons = np.random.randint(1,total_polygons)
		centers = []
		radii = []
		polygons = []
		for p in range(num_polygons):
			radius = abs(40 + 40*np.random.rand())
			overlap = True
			while(overlap):
				c_x = 2*radius + (500-3*radius)*np.random.rand()
				c_y = 2*radius + (500-3*radius)*np.random.rand()
				found = False
				for i in range(len(centers)):
					if params.no_overlap:
						if(distance(centers[i],[c_x,centers[i][1]])>(radii[i]+radius)*1.5):
							continue
						else:
							found = True
							break
					else:						
						if(distance(centers[i],[c_x,c_y])>(radii[i]+radius)*1.5):
							continue
						else:
							found = True
							break
				overlap = found
				if(not overlap):
					centers.append([c_x,c_y])
					radii.append(radius)
			verts = generatePolygon(ctrX=centers[p][0], ctrY=centers[p][1], aveRadius=radii[p])
			num_verts = len(verts)
			max_verts = max(max_verts,num_verts)
			polygons.append(verts)
		writePolygons(f, polygons, pad_token)
		allnormals = writeNormals(f_normal, polygons, pad_token)
		drawPolygons(polygons)#,normals = allnormals)
		#w = input("we")
	f.close()
	f_normal.close()
	f_meta = open(os.path.join(filepath,'meta_%s.dat'%suffix),'w')
	f_meta.write(str(max_verts)+"\n")
	f_meta.write(str(data_size))
	f_meta.close()
def drawPolygons(polygons, proj_pred=None, proj_gt=None, color='red',out='out.png',A=None, line=None):
	black = (0,0,0)
	white=(255,255,255)
	im = Image.new('RGB', (600, 600), white)
	imPxAccess = im.load()
	draw = ImageDraw.Draw(im,'RGBA')
	verts = polygons
	
	# either use .polygon(), if you want to fill the area with a solid colour
	points = tuple(tuple(x) for x in verts)
	#draw.point((points),fill=(255,0,0,0))
	i = 0
	#draw.point((points),fill=(255,0,0,0))
	for points in polygons:
		for point in points:
		    draw.ellipse((point[0] - 4, point[1] - 4, point[0]  + 4, point[1] + 4), fill='green')
		draw.polygon((points), outline='red',fill=(0,0,0,0))

	
	im.save(out)

if __name__ == '__main__':
    args = parseArgs()
    print(args)
    dataGenerator(args)