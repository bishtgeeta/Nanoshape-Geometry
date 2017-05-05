import numpy
import matplotlib.pyplot as plt

def calculate_van_der_waal(A, ps ,r):
    Vdw = A/6 * ( (2*ps**2 / (r**2 - 4*ps**2) ) +  ( 2*ps**2/r**2 ) +  numpy.log( (r**2 - 4*ps**2 ) / r**2) )
    Vdw /= (kB*T)
    return Vdw


def calculate_coulombpotential(r):
    U = (i_4PiEps*e**2/(eps0*r) * numpy.exp(-kappa*(r-sigma))/(1+kappa*sigma) / (kB*T))
    return U

conc = 1e-3
kappa = 1/(0.304/numpy.sqrt(conc)*1e-9)
e = 1.6e-19
i_4PiEps = 9e9
eps0 = 81
kB = 1.38e-23
T = 300
A = 40e-20    
ps = 0.5e-9
sigma = 2*ps
distances = numpy.linspace(2*ps + 1e-10, 10e-9, 100)

vdw_potentials = []
coulomb_potentials = []
for r in distances:
    vdw_potentials.append(calculate_van_der_waal(A, ps ,r))
    coulomb_potentials.append(calculate_coulombpotential(r))

plt.figure()    
plt.plot(distances * 1e9, vdw_potentials)
plt.yscale('log')
plt.ylabel('van der waal energy (kBT)')
plt.xlabel('distance between point particles (nm)')
plt.savefig('vdw.png', dpi=300)

plt.figure()
plt.plot(distances * 1e9, coulomb_potentials)
plt.yscale('log')
plt.ylabel('coloumb energy (kBT)')
plt.xlabel('distance between point particles (nm)')
plt.savefig('coloumb.png', dpi=300)
