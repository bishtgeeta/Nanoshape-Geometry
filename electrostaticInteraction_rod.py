import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from tqdm import tqdm
import copy

class Rod(object):
    def __init__(self,x,y,z,R,H,mesh_size):
        L = W = 2*R
        dx = dy = dz = mesh_size
        Nx = int(L / mesh_size)
        Ny = int(W / mesh_size)
        Nz = int(H / mesh_size)
        self.mesh_size = mesh_size
        
        self.center = numpy.array([x + L/2., y + W/2., z + H/2.])
        self.max_height = H
        self.radius = R
        
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
                    if self.within_rod(point):
                        allPoints[point_counter] = point
                        point_counter += 1
                    
        allPoints = allPoints[:point_counter] 
        self.allPointsDict['allPoints'] = allPoints
        self.x_extent, self.y_extent, self.z_extent = allPoints.max(axis=0) - allPoints.min(axis=0) + self.mesh_size
        
        vertex_counter = edge_counter = face_counter = inner_counter = 0
        print "Creating a rod "
        for point in tqdm(allPoints):
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
                totalNeighbors += (_d < self.mesh_size*1e-2).sum()
                        
            if (totalNeighbors <= 3):
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
    
    
    def within_rod(self, point):
        vector_from_center = point - self.center
        height_from_center = numpy.abs(vector_from_center[2])
        if height_from_center > self.max_height/2.:
            return False
        
        if (numpy.linalg.norm(vector_from_center[:2]) - self.radius > self.mesh_size*1e-2):
            return False
            
        return True

        
    def visualize(self):
		#~ vertex_points = self.allPointsDict['vertex']
		#~ edge_points = self.allPointsDict['edge']
		#~ face_points = self.allPointsDict['face']
		#~ inner_points = self.allPointsDict['inner']
		#~ outer_points = numpy.row_stack((vertex_points, edge_points, face_points))
		#~ xi,yi,zi = inner_points.T
		#~ xo,yo,zo = outer_points.T
		#~ maya.points3d(xo,yo,zo,color=(0.2,0.2,0.8),scale_factor=1)
		#~ maya.points3d(xi,yi,zi,color=(0.8,0.2,0),scale_factor=0.5)
        
        fig = plt.figure()
        ax = Axes3D(fig)
        vertex_points = self.allPointsDict['vertex']
        edge_points = self.allPointsDict['edge']
        face_points = self.allPointsDict['face']
        inner_points = self.allPointsDict['inner']
        outer_points = numpy.row_stack((vertex_points, edge_points, face_points))
        xi,yi,zi = inner_points.T
        xo,yo,zo = outer_points.T
        ax.scatter(xi,yi,zi,color='steelblue')
        ax.scatter(xo,yo,zo,color='orangered', c='orangered')
        plt.savefig(r'Z:\Geeta-Share\rod assembly\interaction potential\rod_geometry(final-0.5nm).png', dpi=300)
             
    def shift(self, d):
        new = copy.deepcopy(self)
        new.center = new.center + d
        for key in new.allPointsDict.keys():
            new.allPointsDict[key] = new.allPointsDict[key] + d
        
        return new



def interactionPotential(rod1,rod2):
    U = 0
    conc = 1e-3
    kappa = 1/(0.304/numpy.sqrt(conc)*1e-9)
    sigma = rod1.mesh_size*1e-9 ## dx, dy, dz are equal to mesh_size
    
    e = 1.6e-19
    i_4PiEps = 9e9
    eps0 = 81
    kB = 1.38e-23
    T = 300
    A = 40e-20

    for key1 in rod1.allPointsDict.keys():
        if key1 == 'allPoints':
            continue
        w1 = rod1.weights[key1]
        if w1 == 0:
            continue
        point1 = rod1.allPointsDict[key1]
        for key2 in rod2.allPointsDict.keys():
            if key2 == 'allPoints':
                continue
            w2 = rod2.weights[key2]
            if w2 == 0:
                continue
            
            point2 = rod2.allPointsDict[key2]
            distance_vector = numpy.dstack((numpy.subtract.outer(point1[:,i], point2[:,i]) for i in range(3)))
            r = numpy.linalg.norm(distance_vector, axis=-1)*1e-9
            
            U += (i_4PiEps*e**2/(eps0*r) * numpy.exp(-kappa*(r-sigma))/(1+kappa*sigma) / (kB*T)).sum()
            
    ## calculation of van der waals potential        
    points1 = rod1.allPointsDict['allPoints']
    points2 = rod2.allPointsDict['allPoints']
    distance_vector = numpy.dstack((numpy.subtract.outer(points1[:,i], points2[:,i]) for i in range(3)))
    r = numpy.linalg.norm(distance_vector, axis=-1)*1e-9
    ps = sigma * 0.49
    Vdw = A/6 * ( (2*ps**2 / (r**2 - 4*ps**2) ) +  ( 2*ps**2/r**2 ) +  numpy.log( (r**2 - 4*ps**2 ) / r**2) ).sum()
    Vdw /= (kB*T)
    print r.min()
    
    return U,Vdw
    
    


start = time.time()
timeList,dList = [],numpy.concatenate((numpy.linspace(0,10,101),range(11,101)))
Uside2sideArray, Utip2tipArray = numpy.zeros((len(dList), 2)), numpy.zeros((len(dList), 2))

outFile1 = open(r'Z:\Geeta-Share\rod assembly\interaction potential\interactionPotential_s2s_rod(final-0.5nm).dat', 'w')
outFile2 = open(r'Z:\Geeta-Share\rod assembly\interaction potential\interactionPotential_t2t_rod(final-0.5nm).dat', 'w')
outFile1.write("Separation Potential\n")
outFile2.write("Separation Potential\n")

mesh_size = 0.5
rod1 = Rod(0,0,0,5,34,mesh_size)
x_extent = rod1.x_extent
z_extent = rod1.z_extent

print "Side by side"
for n,d in tqdm(enumerate(dList)):
    d_vector = numpy.array([x_extent + d, 0, 0])
    rod2 = rod1.shift(d_vector)
    U,Vdw = interactionPotential(rod1,rod2)
    Uside2sideArray[n] = [U,Vdw]
    outFile1.write("%f %.18e %.18e\n" %(d,U,Vdw))

print "Tip to tip"
for n,d in tqdm(enumerate(dList)):
    d_vector = numpy.array([0, 0, z_extent + d])
    rod2 = rod1.shift(d_vector)
    U,Vdw = interactionPotential(rod1,rod2)
    Utip2tipArray[n] = [U,Vdw]
    outFile2.write("%f %.18e %.18e\n" %(d,U,Vdw))

end = time.time()
print "Total number of runs = ", 2*len(dList)
print "Time for full run = {0} minutes".format((end-start) / 60.)
print "Time for each run = {0} minutes".format((end-start)/(60. * len(dList)))

outFile1.close()
outFile2.close()

fig, (ax1, ax2) = plt.subplots(2, figsize=(5,7))
ax1.plot(dList, Uside2sideArray[:,0], color='steelblue')
ax1.plot(dList, Utip2tipArray[:,0], color='orangered')
ax1.set_yscale('log')
ax1.set_ylabel('Coulomb potential')
ax1.legend(('side to side', 'tip to tip'), frameon=False)

ax2.plot(dList, Uside2sideArray[:,1], color='steelblue')
ax2.plot(dList, Utip2tipArray[:,1], color='orangered')
ax2.set_yscale('log')
ax2.set_ylabel('Van der waal potential')
ax2.legend(('side to side', 'tip to tip'), frameon=False)

plt.xlabel('distance between rods (nm)')
plt.tight_layout()
plt.savefig(r'Z:\Geeta-Share\rod assembly\interaction potential\rod_potentials(final-0.5nm).png', dpi=300)
rod1.visualize()
plt.show()

print "number of points in rod1 :"
for key, value in rod1.allPointsDict.items():
    print key, value.shape[0]
 
print rod1.allPointsDict['vertex'].shape[0] + rod1.allPointsDict['edge'].shape[0] + rod1.allPointsDict['face'].shape[0]
