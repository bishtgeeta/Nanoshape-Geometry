import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from tqdm import tqdm
import copy

class BiPyramid(object):
    def __init__(self,x,y,z,R,H,Nx,Ny,Nz):
        L = W = 2*R
        dx,dy,dz = 1.0*L/Nx, 1.0*W/Ny, 1.0*H/Nz
        self.point_size = numpy.linalg.norm([dx,dy,dz])
        
        self.center = numpy.array([x + L/2., y + W/2., z + H/2.])
        self.max_height = H
        self.max_radius = R
        
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
        print "Creating a bipyramid "
        for point in tqdm(allPoints[:point_counter]):
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
                _d = numpy.linalg.norm(neighbor - allPoints[:point_counter], axis=1)
                totalNeighbors += (_d < self.point_size*1e-2).sum()
                        
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
    
    
    def within_bipyramid(self, point):
        vector_from_center = point - self.center
        height_from_center = numpy.abs(vector_from_center[2])
        if height_from_center > self.max_height/2.:
            return False
        
        ## reduction factor tells us how much length and width 
        ## decreases as we move away from center
        reduction_factor = (self.max_height - 2*height_from_center) / self.max_height
        
        central_angle = numpy.deg2rad(36)
        ## _orientation is angle of point from x-axis
        ## _theta is angle of point from nearest vertex
        _orientation = numpy.arccos(vector_from_center[0] / numpy.linalg.norm(vector_from_center[:2]))
        _theta = numpy.abs(_orientation) % (2*central_angle)
        _extent = self.max_radius * numpy.cos(central_angle) / numpy.cos((central_angle - _theta))
        _extent *= reduction_factor
        
        if (numpy.linalg.norm(point[:2] - self.center[:2]) - _extent) > self.point_size*1e-2:
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
        plt.savefig(r'Z:\Geeta-Share\bipyramid assembly\interaction potential\bipyramid_geometry(final-1nm).png', dpi=300)
             
    def shift(self, d):
        new = copy.deepcopy(self)
        for key in new.allPointsDict.keys():
            new.allPointsDict[key] = new.allPointsDict[key] + d
        
        return new
        
        
def interactionPotential(rod1,rod2):
    U = 0
    conc = 1e-3
    kappa = 1/(0.304/numpy.sqrt(conc)*1e-9)
    sigma = rod1.point_size*1e-9 / numpy.sqrt(3)  ## ASSUMPTION : dx, dy, dz are equal
                                             ## for comparision with sphere, anyway
                                             ## dx, dy, dz should be equal
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
    distance_vector = numpy.dstack((numpy.subtract.outer(point1[:,i], point2[:,i]) for i in range(3)))
    r = numpy.linalg.norm(distance_vector, axis=-1)*1e-9
    ps = sigma * 0.49
    Vdw = A/6 * ( (2*ps**2 / (r**2 - 4*ps**2) ) +  ( 2*ps**2/r**2 ) +  numpy.log( (r**2 - 4*ps**2 ) / r**2) ).sum()
    Vdw /= (kB*T)
    print r.min()

    return U,Vdw
    
    
start = time.time()
timeList,dList = [],numpy.concatenate((numpy.linspace(0,10,101),range(11,101)))
Uside2sideArray, Utip2tipArray = numpy.zeros((len(dList), 2)), numpy.zeros((len(dList), 2))

outFile1 = open(r'Z:\Geeta-Share\bipyramid assembly\interaction potential\interactionPotential_s2s_bp(final-1nm).dat', 'w')
outFile2 = open(r'Z:\Geeta-Share\bipyramid assembly\interaction potential\interactionPotential_t2t_bp(final-1nm).dat', 'w')
outFile1.write("Separation Potential\n")
outFile2.write("Separation Potential\n")

bp1 = BiPyramid(0,0,0,10,55,20,20,55)

x_extent = bp1.allPointsDict['allPoints'].max(axis=0)[0] - bp1.allPointsDict['allPoints'].min(axis=0)[0] + 1
z_extent = bp1.allPointsDict['allPoints'].max(axis=0)[2] - bp1.allPointsDict['allPoints'].min(axis=0)[2] + 1
## TODO - add x_extent, y_extent, and z_extent as attributes within the class

print "Side by side"
for n,d in tqdm(enumerate(dList)):
    d_vector = numpy.array([x_extent + d, 0, 0])
    bp2 = bp1.shift(d_vector)
    U,Vdw =  interactionPotential(bp1,bp2)
    Uside2sideArray[n] = [U,Vdw]
    outFile1.write("%f %.18e %.18e\n" %(d,U,Vdw))

print "Tip to tip"
for n,d in tqdm(enumerate(dList)):
    d_vector = numpy.array([0, 0, z_extent + d])
    bp2 = bp1.shift(d_vector)
    U,Vdw = interactionPotential(bp1,bp2)
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

plt.xlabel('distance between bipyramids (nm)')
plt.tight_layout()
plt.savefig(r'Z:\Geeta-Share\bipyramid assembly\interaction potential\bipyramid_potentials(final-1nm).png', dpi=300)
bp1.visualize()
plt.show()


print "number of points in bp1 :"
for key, value in bp1.allPointsDict.items():
    print key, value.shape[0]
 
print bp1.allPointsDict['vertex'].shape[0] + bp1.allPointsDict['edge'].shape[0] + bp1.allPointsDict['face'].shape[0]
