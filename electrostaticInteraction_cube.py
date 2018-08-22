import numpy
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import copy
import os
from os.path import join
from shapes import Cube
from utils import interactionPotential, hydrophobic_potential

conc = 1e-7
A = 10e-20

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
    U,Vdw =  interactionPotential(cube1, cube2,conc,A)
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
