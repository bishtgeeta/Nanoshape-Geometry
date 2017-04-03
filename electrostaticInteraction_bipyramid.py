import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from tqdm import tqdm


class BiPyramid(object):
    def __init__(self,x,y,z,L,W,H,Nx,Ny,Nz):
        dx,dy,dz = 1.0*L/Nx,1.0*W/Ny,1.0*H/Nz
        eps = 1e-10
        
        self.center = numpy.array([x + L/2., y + W/2., z + H/2.])
        self.max_length = L
        self.max_width = W
        self.max_height = H
        
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
                    if self.within_bipyramid(point):
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
    
    
    def within_bipyramid(self, point):
        height_from_center = numpy.abs(point[2] - self.center[2])
        if height_from_center > self.max_height/2.:
            return False
        
        ## reduction factor tells us how much length and width 
        ## decreases as we move away from center
        reduction_factor = (self.max_height - 2*height_from_center) / self.max_height
        
        # length at this distance from center
        _length = self.max_length * reduction_factor
        if numpy.abs(point[0] - self.center[0]) > _length/2.:
            return False
            
        # width at this distance from center
        _width = self.max_width * reduction_factor
        if numpy.abs(point[1] - self.center[1]) > _width/2.:
            return False

        return True
        
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
Uside2sideList,Utip2tipList = [],[]

outFile1 = open('interactionPotential_s2s_1bp.dat', 'w')
outFile2 = open('interactionPotential_t2t_1bp.dat', 'w')
outFile1.write("Separation Potential\n")
outFile2.write("Separation Potential\n")

bp1 = BiPyramid(0,0,0,15,15,50,15,15,50)

print "Side by side"
for d in tqdm(dList):
    bp2 = BiPyramid(15+d,0,0,15,15,50,15,15,50)
    U =  interactionPotential(bp1,bp2)
    Uside2sideList.append(U)
    outFile1.write("%f %f\n" %(d,U))
print "Tip to tip"
for d in tqdm(dList):
    bp2 = BiPyramid(0,0,50+d,15,15,50,15,15,50)
    U = interactionPotential(bp1,bp2)
    Utip2tipList.append(U)
    outFile2.write("%f %f\n" %(d,U))
end = time.time()
print "Total number of runs = ", 2*len(dList)
print "Time for full run = {0} minutes".format((end-start) / 60.)
print "Time for each run = {0} minutes".format((end-start)/(60. * len(dList)))

outFile1.close()
outFile2.close()
#ratio1,ratio2 = [],[]
#for i,j in zip(Uside2sideList,Utip2tipList):
    #ratio1.append(i/j)
    #ratio2.append(j/i)
    
fig = plt.figure(figsize=(4,3))
ax = fig.add_axes([0,0,1,1])
ax.plot(dList,Uside2sideList)
ax.plot(dList,Utip2tipList)
ax.set_xlabel('d')
ax.set_ylabel('U')
plt.savefig('InteractionPotentials_bp.png', dpi=300)

#~ plt.close()