import numpy
import mayavi.mlab as maya
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from os.path import join
from shapes import BiPyramid5f, Sheet
from utils import interactionPotential

use_mayavi = True
bp_height = 40 ## length along long axis
bp_radius = 12  ## distance of pentagon vertex from center
mesh_size = 1
bp = BiPyramid5f(0,0,0,bp_radius,bp_height,mesh_size)
tan_angle = bp_height / (2*bp_radius*numpy.cos(numpy.deg2rad(36)))
angle = numpy.arctan(tan_angle) + numpy.pi
bp = bp.rotate('y', angle)
sheet = Sheet(-20,-20,0,50,50,1,mesh_size)

d = bp.get_extent()[2][0] - sheet.get_extent()[2][0] ## current distance between bp and sheet
bp = bp.shift([0, 0, -d+mesh_size])  ## shift bipyramid close to sheet

bp.visualize(use_mayavi)
sheet.visualize(use_mayavi)
if use_mayavi:
    maya.show()
else:
    plt.show()


root = r'Z:\Geeta-Share\bipyramid5f_sheet\interaction potential'
name = 'bp_sheet'

start = time.time()
timeList,dList = [],numpy.concatenate((numpy.linspace(0,10,101),range(11,101)))
Uside2sideArray = numpy.zeros((len(dList), 2))

outFile1 = open(join(root, 'interactionPotential_s2s_{0}(final-{1}nm).dat'.format(name, mesh_size)), 'w')
outFile1.write("Separation Potential\n")


for n,d in tqdm(enumerate(dList)):
    d_vector = numpy.array([0, 0, d])
    new_bp = bp.shift(d_vector)
    U,Vdw =  interactionPotential(new_bp, sheet)
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

plt.xlabel('distance between bipyramid and sheet (nm)')
plt.tight_layout()
plt.savefig(join(root, '{0}_potential(final-{1}nm).png'.format(name, mesh_size)), dpi=300)
plt.show()


print "number of points in bp :"
for key, value in bp.allPointsDict.items():
    print key, value.shape[0]
