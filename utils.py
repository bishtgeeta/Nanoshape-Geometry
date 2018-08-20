import numpy

def interactionPotential(shape1,shape2):
    U = 0
    conc = 1e-3
    kappa = 1/(0.304/numpy.sqrt(conc)*1e-9)
    sigma = shape1.mesh_size*1e-9  ## dx, dy, dz are equal to mesh_size

    e = 1.6e-19
    i_4PiEps = 9e9
    eps0 = 81
    kB = 1.38e-23
    T = 300
    A = 40e-20
    z = (3.09e-24)**2

    for key1 in shape1.allPointsDict.keys():
        if (key1 == 'allPoints'):
            continue
        w1 = shape1.weights[key1]
        if w1 == 0:
            continue
        point1 = shape1.allPointsDict[key1]
        for key2 in shape2.allPointsDict.keys():
            if (key2 == 'allPoints'):
                continue
            w2 = shape2.weights[key2]
            if w2 == 0:
                continue
            
            point2 = shape2.allPointsDict[key2]
            distance_vector = numpy.dstack((numpy.subtract.outer(point1[:,i], point2[:,i]) for i in range(3)))
            r = numpy.linalg.norm(distance_vector, axis=-1)*1e-9
            
            U += (z*i_4PiEps/(eps0*r) * numpy.exp(-kappa*(r-sigma))/(1+kappa*sigma) / (kB*T)).sum()

    ## calculation of van der waals potential        
    points1 = shape1.allPointsDict['allPoints']
    points2 = shape2.allPointsDict['allPoints']
    distance_vector = numpy.dstack((numpy.subtract.outer(points1[:,i], points2[:,i]) for i in range(3)))
    r = numpy.linalg.norm(distance_vector, axis=-1)*1e-9
    ps = sigma * 0.499
    Vdw = A/6 * ( (2*ps**2 / (r**2 - 4*ps**2) ) +  ( 2*ps**2/r**2 ) +  numpy.log( (r**2 - 4*ps**2 ) / r**2) ).sum()
    Vdw /= (kB*T)
    #~ print r.min(), (r == r.min()).sum()

    return U,Vdw
    
def hydrophobic_potential(d, Dh):
    gamma = 50e-3
    Hy = 0.2
    kB = 1.38e-23
    T = 300
    A = 9e-16
    
    hydrophobic_potential = 2 * gamma * Hy * numpy.exp(-d/Dh) * A / (kB*T)
    return hydrophobic_potential
