import math, random
from PIL import Image, ImageDraw
import numpy as np
import argparse
import os
#from src.util import utils

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

def generatePolygon( ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts) :
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
	irregularity = clip( irregularity, 0,1 ) * 2*math.pi / numVerts
	spikeyness = clip( spikeyness, 0,1 ) * aveRadius

	# generate n angle steps
	angleSteps = []
	lower = (2*math.pi / numVerts) - irregularity
	upper = (2*math.pi / numVerts) + irregularity
	sum = 0
	for i in range(numVerts) :
		tmp = random.uniform(lower, upper)
		angleSteps.append( tmp )
		sum = sum + tmp

	# normalize the steps so that point 0 and point n+1 are the same
	k = sum / (2*math.pi)
	for i in range(numVerts) :
		angleSteps[i] = angleSteps[i] / k

	# now generate the points
	points = []
	angle = random.uniform(0, 2*math.pi)
	for i in range(numVerts) :
		r_i = clip( random.gauss(aveRadius, spikeyness), 0, 2*aveRadius )
		x = ctrX + r_i*math.cos(angle)
		y = ctrY + r_i*math.sin(angle)
		points.append( (int(x),int(y)) )

		angle = angle + angleSteps[i]

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
		aveRadius = abs(50*np.random.randn())
		centers = []
		radii = []
		polygons = []
		for p in range(num_polygons):
			radius = 40 + 10*np.random.rand()
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
			num_verts = int(np.ceil(abs(params.sigma*np.random.randn())+1e-10)) + 2 #3*(2**np.random.randint(0,4))
			max_verts = max(num_verts,max_verts)
			verts = generatePolygon(ctrX=centers[p][0], ctrY=centers[p][1], aveRadius=radii[p], irregularity=0.35, spikeyness=0.2, numVerts=num_verts)
			polygons.append(verts)
		writePolygons(f, polygons, pad_token)
		allnormals = writeNormals(f_normal, polygons, pad_token)
		#utils.drawPolygons(polygons)#,normals = allnormals)
		#w = input("we")
	f.close()
	f_normal.close()
	f_meta = open(os.path.join(filepath,'meta_%s.dat'%suffix),'w')
	f_meta.write(str(max_verts)+"\n")
	f_meta.write(str(data_size))
	f_meta.close()

if __name__ == '__main__':
    args = parseArgs()
    print(args)
    dataGenerator(args)