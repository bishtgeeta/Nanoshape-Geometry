import numpy
import os
from os.path import join
from tqdm import tqdm


timeList,dList = [],numpy.concatenate((numpy.linspace(0,10,101),range(11,101)))                     
Uside2sideArray = numpy.zeros((len(dList), 2))

root = r'Z:\Geeta-Share\sphere assembly\interaction potential'
name = 'sphere'
mesh_size = 1
outFile1 = open(join(root, 'interactionPotential_{0}(finalWithHP(Hy=0.2)-{1}nm).dat'.format(name, mesh_size)), 'w')
outFile1.write("Separation Potential\n")


def hydrophobic_potential(d, Dh):
    gamma = 50e-3
    Hy =0.2
    kB = 1.38e-23
    T = 300
    A = 7.85e-17
    hydrophobic_potential = 2 * gamma * Hy * numpy.exp(-d/Dh) * A / (kB*T)
    return hydrophobic_potential
    

for n,d in tqdm(enumerate(dList)):
    hp2 = hydrophobic_potential(d, 1)
    hp3 = hydrophobic_potential(d, 2)
    Uside2sideArray[n] = [hp2, hp3]
    outFile1.write("%f %.18e %.18e\n" %(d, hp2, hp3))
