import numpy
import mayavi.mlab as maya
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import copy
from os.path import join
from shapes import BiPyramid, Sheet
from utils import interactionPotential



use_mayavi = True
rod_height = 50 ## or length along long axis
rod_radius = 7.5
mesh_size = 1
rod = Rod(-rod_radius,-rod_radius,0,rod_radius,rod_height,mesh_size)
rod = rod.rotate('x', numpy.pi/2)
sheet = Sheet(-25,-25,0,50,50,1,mesh_size)

d = rod.get_extent()[2][0] - sheet.get_extent()[2][0] ## current distance between rod and sheet
rod = rod.shift([0, 0, -d+mesh_size])  ## shift rod close to sheet

rod.visualize(use_mayavi)
sheet.visualize(use_mayavi)
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
