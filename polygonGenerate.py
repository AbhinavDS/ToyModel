import math, random
from PIL import Image, ImageDraw
import numpy as np
def generatePolygon( ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts ) :
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
	else :             return x

def distance(a,b):
	return np.math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
def drawPolygons(polygons,color='red',out='out.png',normals = None):
	black = (0,0,0)
	white=(255,255,255)
	im = Image.new('RGB', (600, 600), white)
	imPxAccess = im.load()
	draw = ImageDraw.Draw(im,'RGBA')
	for i in range(len(polygons)):
		verts = polygons[i]
		points = tuple(verts)
		if normals is None:
			pass
		else:
			polygonnormals = normals[i]
			for point,normal in zip(points,polygonnormals):
				draw.ellipse((point[0]+10*normal[0] - 4, point[1]+10*normal[1] - 4, point[0]+10*normal[0]  + 4, point[1]+10*normal[1] + 4), fill='blue')
		# either use .polygon(), if you want to fill the area with a solid colour
		#draw.point((points),fill=(255,0,0,0))
		for point in points:
		    draw.ellipse((point[0] - 4, point[1] - 4, point[0]  + 4, point[1] + 4), fill=color)
		draw.polygon((points), outline=black,fill=(0,0,0,0) )

		# # or .line() if you want to control the line thickness, or use both methods together!
		tupVerts = (points)
		#draw.line(tupVerts+(tupVerts[0]), width=2, fill=black )
	im.save(out)
DATA_SIZE = 1
PAD_TOKEN = ', -1,-1,'
def writePolygons(file,polygons):
	for p in range(len(polygons)):
		polygon  = polygons[p]
		for v in range(len(polygon)):
			vert = polygon[v]
			if(v == 0):
				file.write(str(vert[0])+','+str(vert[1]))
			else:
				file.write(','+str(vert[0])+','+str(vert[1]))
		if(p!=len(polygons)-1):
			file.write(PAD_TOKEN)
	file.write('\n')
def normalise(v):
	normal2 = v
	return [normal2[0]/math.sqrt(normal2[1]**2+normal2[0]**2), normal2[1]/math.sqrt(normal2[1]**2+normal2[0]**2)]
def writeNormals(file,polygons):
	allnormals = []
	for p in range(len(polygons)):
		polygon  = polygons[p]
		polygonnormals = []
		for v in range(len(polygon)):
			vertp = polygon[v-1]
			vert = polygon[v]
			vertn = polygon[(v+1)%len(polygon)]
			# vertp = [317,322]
			# vert = [327,310]
			# vertn = [329,302]
			normal1 = [vert[1]-vertp[1],vertp[0]-vert[0]]
			normal1 = normalise(normal1)

			normal2 = [vertn[1]-vert[1],vert[0]-vertn[0]]
			normal2 = normalise(normal2)
			normal = [0,0]
			normal[0] = normal1[0] + normal2[0]
			normal[1] = normal1[1] + normal2[1]
			normal = normalise(normal)
			# print(vertp,vert,vertn,normal1,normal2,normal)
			# w = input("eui")
			if(v == 0):
				file.write(str(normal[0])+','+str(normal[1]))
			else:
				file.write(','+str(normal[0])+','+str(normal[1]))
			polygonnormals.append(normal)
		allnormals.append(polygonnormals)
		if(p!=len(polygons)-1):
			file.write(PAD_TOKEN)			
	file.write('\n')
	return allnormals

f = open('polygons.dat','w')
fn = open('normals.dat','w')
for i in range(DATA_SIZE):
	num_polygons = int(np.ceil(abs(3*np.random.randn())))
	num_polygons = 1
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
				if(distance(centers[i],[c_x,c_y])>(radii[i]+radius)*1.5):
					continue
				else:
					found = True
					break
			overlap = found
			if(not overlap):
				centers.append([c_x,c_y])
				radii.append(radius)
		num_verts = 3*(2**np.random.randint(0,4))#int(np.ceil(abs(10*np.random.randn())))+2
		verts = generatePolygon(ctrX=centers[p][0], ctrY=centers[p][1], aveRadius=radii[p], irregularity=0.35, spikeyness=0.2, numVerts=num_verts)
		polygons.append(verts)
	writePolygons(f,polygons)
	allnormals = writeNormals(fn,polygons)
	drawPolygons(polygons)#,normals = allnormals)
	#w = input("we")
