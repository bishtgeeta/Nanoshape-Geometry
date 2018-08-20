import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from tqdm import tqdm
import copy
from os.path import join
from shapes import Sphere
from utils import interactionPotential



root = r'Z:\Geeta-Share\sphere assembly\interaction potential'
name = 'sphere'
mesh_size = 1
sphere1 = Sphere(0,0,0,10,mesh_size)
x_extent = sphere1.x_extent
z_extent = sphere1.z_extent

start = time.time()
timeList,dList = [],numpy.concatenate((numpy.linspace(0,10,101),range(11,101)))
Uside2sideArray = numpy.zeros((len(dList), 2))

outFile1 = open(join(root, 'interactionPotential_{0}(finalzeta-{1}nm).dat'.format(name, mesh_size)), 'w')
outFile1.write("Separation Potential\n")

print "Sphere ..."
for n,d in tqdm(enumerate(dList)):
    d_vector = numpy.array([0,0,z_extent + d])
    sphere2 = sphere1.shift(d_vector)
    U,Vdw =  interactionPotential(sphere1, sphere2)
    Uside2sideArray[n] = [U,Vdw]
    outFile1.write("%f %.18e %.18e\n" %(d,U,Vdw))

end = time.time()
print "Total number of runs = ", len(dList)
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

plt.xlabel('distance between spheres (nm)')
plt.tight_layout()
plt.savefig(join(root, '{0}_potential(finalzeta-{1}nm).png'.format(name, mesh_size)), dpi=300)
sphere1.visualize()
plt.savefig(join(root, '{0}_geometry(finalzeta-{1}nm).png'.format(name, mesh_size)), dpi=300)
plt.show()

print "number of points in sphere1:"
for key, value in sphere1.allPointsDict.items():
    print key, value.shape[0]
 
print sphere1.allPointsDict['vertex'].shape[0] + sphere1.allPointsDict['edge'].shape[0] + sphere1.allPointsDict['face'].shape[0]
