import numpy
import mayavi.mlab as maya

def interactionPotential(shape1,shape2,conc,A,T=300,z=1):
    U = 0
    kappa = 1/(0.304/numpy.sqrt(conc)*1e-9)
    sigma = shape2.mesh_size*1e-9  ## dx, dy, dz are equal to mesh_size
    R1 = shape1.mesh_size*1e-9*0.499
    R2 = shape2.mesh_size*1e-9*0.499

    e = 1.6e-19
    i_4PiEps = 9e9
    eps0 = 81
    kB = 1.38e-23
    #z = (3.09e-24)**2

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
            
            U += (i_4PiEps*(z*e)**2/(eps0*r) * numpy.exp(-kappa*(r-sigma))/(1+kappa*sigma) / (kB*T)).sum()

    ## calculation of van der waals potential        
    points1 = shape1.allPointsDict['allPoints']
    points2 = shape2.allPointsDict['allPoints']
    distance_vector = numpy.dstack((numpy.subtract.outer(points1[:,i], points2[:,i]) for i in range(3)))
    r = numpy.linalg.norm(distance_vector, axis=-1)*1e-9
    r_square = r**2
    term1 = r_square - (R1+R2)**2
    term2 = r_square - (R1-R2)**2
    Vdw = A/6 * ( 2*R1*R2 * ( 1/term1 + 1/term2 ) + numpy.log(term1/term2) ).sum()
    Vdw /= (kB*T)
    #~ print r.min(), (r == r.min()).sum()

    return U,Vdw
    
#def hydrophobic_potential(d, Dh):
    #gamma = 50e-3
    #Hy = 0.2
    #kB = 1.38e-23
    #T = 300
    #A = 9e-16
    
    #hydrophobic_potential = 2 * gamma * Hy * numpy.exp(-d/Dh) * A / (kB*T)
    #return hydrophobic_potential
    
    
@maya.animate(delay=100)
def animate(fig, xs, ys, zs):
    while True:
        for (x, y, z) in zip(xs, ys, zs):
            fig.mlab_source.set(x=x, y=y, z=z)
            fig.scene.render()
            yield
