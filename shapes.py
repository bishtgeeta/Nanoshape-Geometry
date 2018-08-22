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
        print "Creating a {}".format(self.shape_name)
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
        
    def visualize(self, use_mayavi=True):
        vertex_points = self.allPointsDict['vertex']
        edge_points = self.allPointsDict['edge']
        face_points = self.allPointsDict['face']
        inner_points = self.allPointsDict['inner']
        outer_points = numpy.row_stack((vertex_points, edge_points, face_points))
        xi,yi,zi = inner_points.T
        xo,yo,zo = outer_points.T
        if use_mayavi:
            maya.points3d(xo,yo,zo,color=(0.2,0.2,0.8),scale_factor=1)
            maya.points3d(xi,yi,zi,color=(0.8,0.2,0),scale_factor=0.5)
        else:
            fig = plt.figure()
            ax = Axes3D(fig)
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
    '''The base of this bypyramid is a square
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
        
        
class BiPyramid5f(Shape):
    '''The base of this bypyramid is a pentagon
    inputs are coordinates of a corner of enclosing cuboid (x,y,z),
    radius of circumcircle of base (R), height of the enclosing 
    cuboid (H), and mesh_size
    '''
    def __init__(self,x,y,z,R,H, mesh_size):
        L = W = 2*R
        self.max_radius = R
        self.max_height = H
        self.shape_name = 'bipyramid (5 faces)'
        super(BiPyramid5f, self).__init__(x,y,z,L,W,H,mesh_size)
        
    
    def within_shape(self, point):
        vector_from_center = point - self.center
        height_from_center = numpy.abs(vector_from_center[2])
        if height_from_center > self.max_height/2.:
            return False
        if numpy.linalg.norm(vector_from_center[:2]) == 0:
            return True
        
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
        
        ## check if the length of vector from center minus _extent is greater than zero
        ## we use mesh_size*1e-6 instead of zero to account for floating point errors
        if numpy.linalg.norm(point[:2] - self.center[:2]) > _extent:
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


class Sphere(Shape):
    def __init__(self,x,y,z,R,mesh_size):
        L = W = H = R*2
        self.radius = R
        self.shape_name = 'sphere'
        super(Sphere, self).__init__(x,y,z,L,W,H,mesh_size)
        
    def within_shape(self, point):
        return numpy.linalg.norm(self.center - point) <= self.radius
        
class Point(Shape):
    
    def __init__(self, x, y, z, mesh_size):
        L = W = H = mesh_size
        super(Point, self).__init__(x, y, z, L, W, H, mesh_size)
        
    def within_shape(self, point):
        return True
        
