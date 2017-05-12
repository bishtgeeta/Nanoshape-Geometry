import numpy
import matplotlib.pyplot as plt

def calculate_van_der_waal(A, ps ,r):
    Vdw = A/6 * ( (2*ps**2 / (r**2 - 4*ps**2) ) +  ( 2*ps**2/r**2 ) +  numpy.log( (r**2 - 4*ps**2 ) / r**2) )
    Vdw /= (kB*T)
    return Vdw

conc = 1e-3
kappa = 1/(0.304/numpy.sqrt(conc)*1e-9)
e = 1.6e-19
i_4PiEps = 9e9
eps0 = 81
kB = 1.38e-23
T = 300
A = 40e-20    
ps = 0.499e-9
sigma = 1e-9
distances = numpy.arange(20, 41, 1)

Vdw = []

for r in distances:
    r = r*1e-9
    Vdw.append(calculate_van_der_waal(A, ps ,r))
   
results = numpy.column_stack((distances-1, U, Vdw))
numpy.savetxt(r'Z:\Geeta-Share\point charge potential\whole_sphere.txt', results, header='distance, van der waal', delimiter= '  ')

plt.figure()    
plt.plot(distances - 1, Vdw)
plt.yscale('log')
plt.ylabel('van der waal energy (kBT)')
plt.xlabel('distance between point particles (nm)')
plt.savefig(r'Z:\Geeta-Share\point charge potential\whole_sphere.png', dpi=300)
