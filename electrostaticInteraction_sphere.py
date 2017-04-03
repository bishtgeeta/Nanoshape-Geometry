import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from tqdm import tqdm

class Sphere(object):
    def __init__(self,x,y,z,R,Nd):
        ## make a cuboid first, and remove extra bits from it.
        L = W = H = R*2
        Nx = Ny = Nz = Nd
        dx,dy,dz = 1.0*L/Nx,1.0*W/Ny,1.0*H/Nz
        eps = 1e-10
        
        self.radius = R
        self.center = numpy.array([x + R, y + R, z + R])
        
        self.allPointsDict = {}
        self.allPointsDict['vertex'] = numpy.zeros((Nx*Ny*Nz, 3))
        self.allPointsDict['edge'] = numpy.zeros((Nx*Ny*Nz, 3))
        self.allPointsDict['face'] = numpy.zeros((Nx*Ny*Nz, 3))
        self.allPointsDict['inner'] = numpy.zeros((Nx*Ny*Nz, 3))
        
        allPoints = numpy.zeros((Nx*Ny*Nz, 3))
        point_counter = 0
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    point = [x+dx/2+i*dx,y+dy/2+j*dy,z+dz/2+k*dz]
                    if self.within_sphere(point):
                        allPoints[point_counter] = point
                        point_counter += 1
                    
        self.allPointsDict['allPoints'] = allPoints[:point_counter]
        vertex_counter = edge_counter = face_counter = inner_counter = 0
        for point in allPoints:
            allNeighbors = numpy.array([
						[point[0]-dx,point[1],point[2]],
						[point[0]+dx,point[1],point[2]],
						[point[0],point[1]+dy,point[2]],
						[point[0],point[1]-dy,point[2]],
						[point[0],point[1],point[2]+dz],
						[point[0],point[1],point[2]-dz]
						])
            totalNeighbors = 0
            for neighbor in allNeighbors:
                _d = numpy.linalg.norm(neighbor - allPoints, axis=1)
                totalNeighbors += (_d < eps).sum()
                        
            if (totalNeighbors == 3):
                self.allPointsDict['vertex'][vertex_counter] = point
                vertex_counter += 1
            elif (totalNeighbors == 4):
                self.allPointsDict['edge'][edge_counter] = point
                edge_counter += 1
            elif (totalNeighbors == 5):
                self.allPointsDict['face'][face_counter] = point
                face_counter += 1
            elif (totalNeighbors == 6):
                self.allPointsDict['inner'][inner_counter] = point
                inner_counter += 1
                
        self.allPointsDict['vertex'] = self.allPointsDict['vertex'][:vertex_counter]
        self.allPointsDict['edge'] = self.allPointsDict['edge'][:edge_counter]
        self.allPointsDict['face'] = self.allPointsDict['face'][:face_counter]
        self.allPointsDict['inner'] = self.allPointsDict['inner'][:inner_counter]
                
        self.weights = {}
        self.weights['vertex'] = 1.0
        self.weights['edge'] = 1.0
        self.weights['face'] = 1.0
        self.weights['inner'] = 0.0
        
    
    def within_sphere(self, point):
        return numpy.linalg.norm(self.center - point) <= self.radius
        
    def visualize(self):
		fig = plt.figure()
		ax = Axes3D(fig)
		vertex_points = self.allPointsDict['vertex']
		edge_points = self.allPointsDict['edge']
		face_points = self.allPointsDict['face']
		inner_points = self.allPointsDict['inner']
		outer_points = numpy.row_stack((vertex_points, edge_points, face_points))
		xi,yi,zi = inner_points.T
		xo,yo,zo = outer_points.T
		#~ ax.scatter(xi,yi,zi,color='steelblue')
		ax.scatter(xo,yo,zo,color='orangered', c='orangered')
		plt.show()
 
         
            
def interactionPotential(rod1,rod2):
    U = 0
    conc = 500e-6
    kappa = 1/(0.152/numpy.sqrt(conc)*1e-9)
    sigma = 5e-9
    e = 1.6e-19
    i_4PiEps = 9e9
    eps0 = 81
    kB = 1.38e-23
    T = 300
    #counter=0
    
    for key1 in rod1.allPointsDict.keys():
        if (key1 != 'allPoints'):
            for point1 in rod1.allPointsDict[key1]:
                x1,y1,z1,w1 = point1[0],point1[1],point1[2],rod1.weights[key1]
                for key2 in rod2.allPointsDict.keys():
                    if (key2 != 'allPoints'):
                        for point2 in rod2.allPointsDict[key2]:
                            x2,y2,z2,w2 = point2[0],point2[1],point2[2],rod2.weights[key2]
                            r = numpy.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)*1e-9
                            
                            U += (w1*w2) * i_4PiEps*e**2/(eps0*r) * numpy.exp(-kappa*(r-sigma))/(1+kappa*sigma) / (kB*T)
                            #if (w1*w2 > 0):
                                #counter+=1
    #print counter
    return U
    
start = time.time()
timeList,dList = [],numpy.concatenate((numpy.linspace(0,10,101),range(11,101)))
Uside2sideList = []

outFile1 = open('interactionPotential_spheres.dat', 'w')
outFile1.write("Separation Potential\n")


sphere1 = Sphere(0,0,0,15,15)

print "Sphere ..."
for d in tqdm(dList):
    sphere2 = Sphere(0,0,30+d,15,30)
    U =  interactionPotential(sphere1, sphere2)
    Uside2sideList.append(U)
    outFile1.write("%f %f\n" %(d,U))

end = time.time()
print "Total number of runs = ", len(dList)
print "Time for full run = {0} minutes".format((end-start) / 60.)
print "Time for each run = {0} minutes".format((end-start)/(60. * len(dList)))

outFile1.close()
    
fig = plt.figure(figsize=(4,3))
ax = fig.add_axes([0,0,1,1])
ax.plot(dList,Uside2sideList)
ax.set_xlabel('d')
ax.set_ylabel('U')
plt.savefig('InteractionPotentials_sphere.png', dpi=300)

plt.close()