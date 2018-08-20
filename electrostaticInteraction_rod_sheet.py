import numpy
import mayavi.mlab as maya
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from tqdm import tqdm
import copy
from os.path import join



class Shape(object):
    shape_name = 'shape'
    def __init__(self,x,y,z,L,W,H,mesh_size):
        dx = dy = dz = mesh_size
        Nx = int(L / mesh_size)
        Ny = int(W / mesh_size)
        Nz = int(H / mesh_size)

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
                    if self.within_shape(point):
                        allPoints[point_counter] = point
                        point_counter += 1

        allPoints = allPoints[:point_counter]
        self.allPointsDict['allPoints'] = allPoints
        self.x_extent, self.y_extent, self.z_extent = allPoints.max(axis=0) - allPoints.min(axis=0) + self.mesh_size

        vertex_counter = edge_counter = face_counter = inner_counter = 0
        # print "Creating a {}".format(self.shape_name)
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
        if use_mayavi:
            vertex_points = self.allPointsDict['vertex']
            edge_points = self.allPointsDict['edge']
            face_points = self.allPointsDict['face']
            inner_points = self.allPointsDict['inner']
            outer_points = numpy.row_stack((vertex_points, edge_points, face_points))
            xi,yi,zi = inner_points.T
            xo,yo,zo = outer_points.T
            maya.points3d(xo,yo,zo,color=(0.2,0.2,0.8),scale_factor=1)
            maya.points3d(xi,yi,zi,color=(0.8,0.2,0),scale_factor=0.5)
        else:
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
        new.center = new.center + numpy.array(d)
        for key in new.allPointsDict.keys():
            new.allPointsDict[key] = new.allPointsDict[key] + d

        return new

    def rotate(self, axis, angle):
        new = self.shift(-self.center)
        sin = numpy.sin(angle)
        cos = numpy.cos(angle)
        if axis == 'x':
            rotation_matrix = [[1,   0,   0],
                               [0, cos, -sin],
                               [0, sin, cos]]
        elif axis == 'y':
            rotation_matrix = [[cos, 0, sin],
                               [0,   1,  0],
                               [-sin, 0, cos]]
        elif axis == 'z':
            rotation_matrix = [[cos, -sin, 0],
                               [sin, cos,  0],
                               [0,   0,    1]]
        else:
            raise ValueError('axis must be one of `x`, `y` or `z`')

        for key, value in new.allPointsDict.items():
            new.allPointsDict[key] = numpy.dot(rotation_matrix, value.T).T

        return new.shift(self.center)

    def get_extent(self):
        allPoints = self.allPointsDict['allPoints']
        xmax, ymax, zmax = allPoints.max(axis=0)
        xmin, ymin, zmin = allPoints.min(axis=0)
        return ((xmin, xmax, xmax-xmin),
                (ymin, ymax, ymax-ymin),
                (zmin, zmax, zmax-zmin))

    def within_shape(self, point):
        raise NotImplementedError('this method must be implemented by '
                                    'a child class of Shape')





class BiPyramid(Shape):
    '''The base of this bypyramid is a square, as opposed to a
    pentagon in the other program
    inputs are coordinates of a corner of enclosing cuboid (x,y,z),
    sides of the square base (s), height of the enclosing cuboid (H),
    and mesh_size
    '''
    def __init__(self,x,y,z,s,H, mesh_size):
        L = W = s
        self.base_side = s
        self.max_height = H
        self.shape_name = 'bipyramid'
        super(BiPyramid, self).__init__(x,y,z,L,W,H,mesh_size)


    def within_shape(self, point):
        vector_from_center = point - self.center
        x_from_center = numpy.abs(vector_from_center[0])
        y_from_center = numpy.abs(vector_from_center[1])
        z_from_center = numpy.abs(vector_from_center[2])
        if z_from_center > self.max_height/2.:
            return False
        if x_from_center == 0 and y_from_center == 0:
            return True

        ## reduction factor tells us how much the side of square base
        ## decreases as we move away from center
        reduction_factor = (self.max_height - 2*z_from_center) / self.max_height

        if x_from_center > reduction_factor * self.base_side/2:
            return False
        if y_from_center > reduction_factor * self.base_side/2:
            return False

        return True

class Sheet(Shape):

    def __init__(self,x,y,z,L,W,H,mesh_size):
        self.shape_name = 'sheet'
        super(Sheet, self).__init__(x,y,z,L,W,H,mesh_size)

    def within_shape(self, point):
        return True


class Rod(Shape):

    def __init__(self,x,y,z,R,H, mesh_size):
        L = W = 2*R
        self.radius = R
        self.max_height = H
        self.shape_name = 'rod'
        super(Rod, self).__init__(x,y,z,L,W,H,mesh_size)


    def within_shape(self, point):
        vector_from_center = point - self.center
        height_from_center = numpy.abs(vector_from_center[2])
        if height_from_center > self.max_height/2.:
            return False
        if numpy.linalg.norm(vector_from_center[:2]) > self.radius:
            return False
        return True



def interactionPotential(rod1,rod2):
    U = 0
    conc = 1e-7    ###Changed value for this code
    kappa = 1/(0.304/numpy.sqrt(conc)*1e-9)
    sigma = rod1.mesh_size*1e-9 ## dx, dy, dz are equal to mesh_size

    e = 1.6e-19
    i_4PiEps = 9e9
    eps0 = 81
    kB = 1.38e-23
    T = 300
    A = 10e-20  ###Changed value for this code

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
    ps = sigma * 0.499
    Vdw = A/6 * ( (2*ps**2 / (r**2 - 4*ps**2) ) +  ( 2*ps**2/r**2 ) +  numpy.log( (r**2 - 4*ps**2 ) / r**2) ).sum()
    Vdw /= (kB*T)

    return U,Vdw



use_mayavi = True
rod_height = 50 ## or length along long axis
rod_radius = 7.5
mesh_size = 1
rod = Rod(-rod_radius,-rod_radius,0,rod_radius,rod_height,mesh_size)
rod = rod.rotate('x', numpy.pi/2)
sheet = Sheet(-25,-25,0,50,50,1,mesh_size)

d = rod.get_extent()[2][0] - sheet.get_extent()[2][0] ## current distance between rod and sheet
rod = rod.shift([0, 0, -d+mesh_size])  ## shift rod close to sheet

rod.visualize()
sheet.visualize()
if use_mayavi:
    maya.show()
else:
    plt.show()


root = r'W:\geeta\Rotation\InteractionPotential\Rod_Sheet'
name = 'rod_sheet'

start = time.time()
timeList,dList = [],numpy.concatenate((numpy.linspace(1,10,91),range(11,101)))
Uside2sideArray = numpy.zeros((len(dList), 2))

outFile1 = open(join(root, 'interactionPotential_s2s_{0}({1}nm).dat'.format(name, mesh_size)), 'w')
outFile1.write("Separation Potential\n")


for n,d in tqdm(enumerate(dList)):
    d_vector = numpy.array([0, 0, d])
    new_rod = rod.shift(d_vector)
    U,Vdw =  interactionPotential(new_rod, sheet)
    Uside2sideArray[n] = [U,Vdw]
    outFile1.write("%f %.18e %.18e\n" %(d,U,Vdw))


end = time.time()
print "Total number of runs = ", 2*len(dList)
print "Time for full run = {0} minutes".format((end-start) / 60.)
print "Time for each run = {0} minutes".format((end-start)/(60. * len(dList)))

outFile1.close()

fig, (ax1, ax2) = plt.subplots(2, figsize=(5,7))
ax1.plot(dList, Uside2sideArray[:,0], color='steelblue')
ax1.set_yscale('log')
ax1.set_ylabel('Coulomb potential')

ax2.plot(dList, Uside2sideArray[:,1], color='steelblue')
ax2.set_yscale('log')
ax2.set_ylabel('Van der waal potential')

plt.xlabel('distance between rod and sheet (nm)')
plt.tight_layout()
plt.savefig(join(root, '{0}_potential({1}nm).png'.format(name, mesh_size)), dpi=300)
plt.show()


print "number of points in rod :"
for key, value in rod.allPointsDict.items():
     print key, value.shape[0]
