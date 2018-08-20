import numpy
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import copy
from os.path import join
from shapes import BiPyramid
from utils import interactionPotential


root = r'Z:\Geeta-Share\bipyramid assembly\interaction potential'
name = 'bp'
mesh_size = 1
bp1 = BiPyramid(0,0,0,10,55,mesh_size)
x_extent = bp1.x_extent
z_extent = bp1.z_extent

start = time.time()
timeList,dList = [],numpy.concatenate((numpy.linspace(0,10,101),range(11,101)))
Uside2sideArray, Utip2tipArray = numpy.zeros((len(dList), 2)), numpy.zeros((len(dList), 2))

outFile1 = open(join(root, 'interactionPotential_s2s_{0}(final-{1}nm).dat'.format(name, mesh_size)), 'w')
outFile2 = open(join(root, 'interactionPotential_t2t_{0}(final-{1}nm).dat'.format(name, mesh_size)), 'w')
outFile1.write("Separation Potential\n")
outFile2.write("Separation Potential\n")


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
plt.savefig(join(root, '{0}_potential(final-{1}nm).png'.format(name, mesh_size)), dpi=300)
bp1.visualize()
plt.savefig(join(root, '{0}_geometry(final-{1}nm).png'.format(name, mesh_size)), dpi=300)
plt.show()


print "number of points in bp1 :"
for key, value in bp1.allPointsDict.items():
    print key, value.shape[0]
 
print bp1.allPointsDict['vertex'].shape[0] + bp1.allPointsDict['edge'].shape[0] + bp1.allPointsDict['face'].shape[0]
