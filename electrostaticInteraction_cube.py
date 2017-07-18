import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from tqdm import tqdm
import copy
import os
from os.path import join

class Cube(object):
    def __init__(self,x,y,z,S,mesh_size):
        L = W = H = S
        Nx = Ny = Nz = int(S / mesh_size)
        dx = dy = dz = mesh_size
        self.mesh_size = mesh_size
        self.center = numpy.array([x + L/2., y + W/2., z + H/2.])
        
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
                    allPoints[point_counter] = point
                    point_counter += 1
                    
        allPoints = allPoints[:point_counter] 
        self.allPointsDict['allPoints'] = allPoints
        self.x_extent, self.y_extent, self.z_extent = allPoints.max(axis=0) - allPoints.min(axis=0) + self.mesh_size
        
        vertex_counter = edge_counter = face_counter = inner_counter = 0
        print "Creating a cube "
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
        ax.scatter(xi,yi,zi,color='steelblue')
        ax.scatter(xo,yo,zo,color='orangered', c='orangered')

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
    sigma = rod1.mesh_size*1e-9  ## dx, dy, dz are equal to mesh_size
    
    e = 1.6e-19
    i_4PiEps = 9e9
    eps0 = 81
    kB = 1.38e-23
    T = 300
    A = 40e-20
    z1 = 0.0144
    z2 = 0.0144

    for key1 in rod1.allPointsDict.keys():
        if (key1 == 'allPoints'):
            continue
        w1 = rod1.weights[key1]
        if w1 == 0:
            continue
        point1 = rod1.allPointsDict[key1]
        for key2 in rod2.allPointsDict.keys():
            if (key2 == 'allPoints'):
                continue
            w2 = rod2.weights[key2]
            if w2 == 0:
                continue
            
            point2 = rod2.allPointsDict[key2]
            distance_vector = numpy.dstack((numpy.subtract.outer(point1[:,i], point2[:,i]) for i in range(3)))
            r = numpy.linalg.norm(distance_vector, axis=-1)*1e-9
            if (w1==0 or w2==0):
                print "here"
            U += (i_4PiEps*z1*z2*e**2/(eps0*r) * numpy.exp(-kappa*(r-sigma))/(1+kappa*sigma) / (kB*T)).sum()

    ## calculation of van der waals potential        
    points1 = rod1.allPointsDict['allPoints']
    points2 = rod2.allPointsDict['allPoints']
    distance_vector = numpy.dstack((numpy.subtract.outer(points1[:,i], points2[:,i]) for i in range(3)))
    r = numpy.linalg.norm(distance_vector, axis=-1)*1e-9
    ps = sigma * 0.499
    Vdw =  A/6 * ( (2*ps**2 / (r**2 - 4*ps**2) ) +  ( 2*ps**2/r**2 ) +  numpy.log( (r**2 - 4*ps**2 ) / r**2) ).sum()
    Vdw /= (kB*T)
    print r.min(), (r == r.min()).sum()
    return U,Vdw
    
def hydrophobic_potential(d, Dh):
    gamma = 50e-3
    Hy = 0.2
    kB = 1.38e-23
    T = 300
    A = 9e-16
    
    hydrophobic_potential = 2 * gamma * Hy * numpy.exp(-d/Dh) * A / (kB*T)
    return hydrophobic_potential
    


root = r'Z:\Geeta-Share\cube assembly\interaction potential'
name = 'cube'
mesh_size = 1
cube1 = Cube(0,0,0,30,mesh_size)
x_extent = cube1.x_extent
z_extent = cube1.z_extent

start = time.time()
timeList,dList = [],numpy.concatenate((numpy.linspace(0,10,101),range(11,101)))                     
Uside2sideArray = numpy.zeros((len(dList), 5))

outFile1 = open(join(root, 'interactionPotential_{0}(finalWith-Z&HP-{1}nm).dat'.format(name, mesh_size)), 'w')
outFile1.write("Separation Potential\n")

print "Cube ..."
for n,d in tqdm(enumerate(dList)):
    d_vector = numpy.array([0,0,z_extent + d])
    cube2 = cube1.shift(d_vector)
    U,Vdw =  interactionPotential(cube1, cube2)
    hp1 = hydrophobic_potential(d, 0.5)
    hp2 = hydrophobic_potential(d, 1)
    hp3 = hydrophobic_potential(d, 2)
    Uside2sideArray[n] = [U, Vdw, hp1, hp2, hp3]
    outFile1.write("%f %.18e %.18e %.18e %.18e %.18e\n" %(d, U, Vdw, hp1, hp2, hp3))

end = time.time()
print "Total number of runs = ", len(dList)
print "Time for full run = {0} minutes".format((end-start) / 60.)
print "Time for each run = {0} minutes".format((end-start)/(60. * len(dList)))

outFile1.close()
    
#fig, (ax1, ax2) = plt.subplots(2, figsize=(5,7))
#ax1.plot(dList, Uside2sideArray[:,0], color='steelblue')
#ax1.set_yscale('log')
#ax1.set_ylabel('Coulomb potential')

#ax2.plot(dList, Uside2sideArray[:,1], color='steelblue')
#ax2.set_yscale('log')
#ax2.set_ylabel('Van der waal potential')

#plt.xlabel('distance between cubes (nm)')
#plt.tight_layout()
#plt.savefig(join(root, '{0}_potential(final-{1}nm).png'.format(name, mesh_size)), dpi=300)
#cube1.visualize()
#plt.savefig(join(root, '{0}_geometry(final-{1}nm).png'.format(name, mesh_size)), dpi=300)
#plt.show()

#print "number of points in cube1 :"
#for key, value in cube1.allPointsDict.items():
    #print key, value.shape[0]
 
#print cube1.allPointsDict['vertex'].shape[0] + cube1.allPointsDict['edge'].shape[0] + cube1.allPointsDict['face'].shape[0]
