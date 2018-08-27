import numpy
import mayavi.mlab as maya
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from os.path import join
from shapes import Point, Sheet
from utils import interactionPotential, animate

conc = 1e-7
A = 10e-20

use_mayavi = True
mesh_size = 1
p = Point(0,0,0,mesh_size=0.1)
sheet = Sheet(0,0,0,50,50,1,mesh_size)
sheet = sheet.shift([-sheet.center[0], -sheet.center[1],-0.55])


p.visualize(use_mayavi)
sheet.visualize(use_mayavi)

if use_mayavi:
    maya.show()
    #~ fig1, fig2 = maya.gcf().children
    ## fig1 is point, fig2 is sheet according to the order in which
    ## `visualize` was called
else:
    plt.show()


root = r'W:\geeta\Rotation\InteractionPotential\point(0.1)_Sheet'
import os
root = os.getcwd()
name = 'point(0.1)_sheet'

start = time.time()
timeList,dList = [],numpy.concatenate((numpy.linspace(0,10,101),range(11,101)))
Uside2sideArray = numpy.zeros((len(dList), 2))

outFile1 = open(join(root, 'interactionPotential{0}({1}nm).dat'.format(name, mesh_size)), 'w')
outFile1.write("Separation Potential\n")


for n,d in tqdm(enumerate(dList)):
    d_vector = numpy.array([0, 0, d])
    new_p = p.shift(d_vector)
    U,Vdw =  interactionPotential(new_p, sheet, conc, A)
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

plt.xlabel('distance between point and sheet (nm)')
plt.tight_layout()
plt.savefig(join(root, '{0}_potential({1}nm).png'.format(name, mesh_size)), dpi=300)
plt.show()


print "number of points in p :"
for key, value in p.allPointsDict.items():
    print key, value.shape[0]
