import numpy as np
from scipy import spatial
import itertools
    
def get_ultracell(fpos,cell,species,r_cut,show=False,verbose=False,max_iter=20):
    if verbose: print("Generating ultracell with r_cut = {}".format(r_cut))
    
    Vcell = np.absolute(np.linalg.det(cell))
    
    # find center and corners of the cell
    center = .5 * cell.sum(axis=1)
    
    fcorners = np.array([[0,0,0],
                         [1,0,0],
                         [0,1,0],
                         [0,0,1],
                         [1,1,0],
                         [1,0,1],
                         [0,1,1],
                         [1,1,1]])
    corners = fcorners.dot(cell)
    
    # plot to make sure
    if show:
        fig = plt.figure()
        ax = Axes3D(fig)
        xco, yco, zco = corners.T
        ax.plot(xco,yco,zco,'dw',alpha=0.4,markeredgecolor="blue",label="corners")
        ax.plot([center[0]],[center[1]],[center[2]],"ro")
        plt.show()
    
    r_center2corner = np.linalg.norm(corners - center,axis=1).max()
    if verbose: print ('rcut = {} rcenter21corner = {}'.format(r_cut,r_center2corner))
    r_search = (r_cut + r_center2corner) * 1.5
    
    Vsphere = 4./3.*np.pi*r_search**3
    if verbose:
        print("approx. num of required cells = {}".format(Vsphere/float(Vcell)))
    
    start = list(itertools.product(*[[-1,0,1] for v in range(3)]))
    ijks_accepted = set(start) # contains all ijks ever accepted
    ijks_to_test = set(start) # contains all ijks which should be tested
    ijks_saturated = set() # contains all ijks which have max number of neighbors
    
    allowed_moves = [v for v in itertools.product(*[[-1,0,1] for v in range(3)]) if not (v[0]==0 and v[1]==0 and v[2]==0)]
    if verbose: print("allowed moves {}".format(allowed_moves))
    
    i = 0
    while i<max_iter:
        if verbose:
            print("\n{}/{}".format(i+1,max_iter))
            print("cells: current = {} estimate for final = {}".format(len(ijks_accepted),Vsphere/float(Vcell)))
        
        # generate possible ijks by going through ijks_to_test comparing to ijks_saturated
        ijks_possible = [(i0+m0,i1+m1,i2+m2) for (i0,i1,i2) in ijks_to_test \
            for (m0,m1,m2) in allowed_moves if (i0+m0,i1+m1,i2+m2) not in ijks_saturated]
        if verbose: print("possible new cells: {}".format(len(ijks_possible)))
        
        # check which ijks are within the specified search radius and add those to ijks_accpeted
        ijks_possible = [(i0,i1,i2) for (i0,i1,i2) in ijks_possible if np.linalg.norm(i0*cell[0,:]+i1*cell[1,:]+i2*cell[2,:])<=r_search]
        if verbose: print("cells after r filter {}".format(len(ijks_possible)))
        if len(ijks_possible) == 0:
            if verbose: print("Found all cells for r_cut {} => r_search = {}, terminating after {} iterations".format(r_cut,r_search,i+1))
            break

        # add all ijks_possible points to ijks_accepted
        ijks_accepted.update(ijks_possible)
        if verbose:print("accepted new cells: {}".format(len(ijks_accepted)))
        
        # all ijks_to_test points now are saturated, hence add to ijks_saturated
        ijks_saturated.update(ijks_to_test)
        if verbose:print("stored cells so far: {}".format(len(ijks_saturated)))
        
        # remove all previously tested points
        ijks_to_test.clear()
        
        # add all points which were not already known to ijks_to_test
        ijks_to_test.update(ijks_possible)
        if verbose:print("cell to test next round: {}".format(len(ijks_to_test)))
        
        i += 1
    if i == max_iter:
        warnings.warn("max_iter reached in the ultracell generation! Generated {}/{} cells. Consider increasing max_iter.".format(len(ijks_accepted),Vsphere/float(Vcell)))
        raise Error("max_iter reached in the ultracell generation! Generated {}/{} cells. Consider increasing max_iter.".format(len(ijks_accepted),Vsphere/float(Vcell)))
    
    # calculating the fractional atom positions
    fbasis = np.eye(3)
    idx_atoms = np.arange(len(fpos))
    
    for h,(i,j,k) in enumerate(ijks_accepted):
        new_fpos = fpos + i*fbasis[0,:] + j*fbasis[1,:] + k*fbasis[2,:]
        if h == 0:
            ultra_fpos = new_fpos
            ultra_species = np.array(species)
            ultracell_idx = idx_atoms
        else:
            ultra_fpos = np.vstack((ultra_fpos,new_fpos))
            ultra_species = np.hstack((ultra_species,species))
            ultracell_idx = np.hstack((ultracell_idx,idx_atoms))
                
    # converting atom positions into umm... non-fractional ...
    ultra_pos = np.dot(ultra_fpos,cell)
    
    return ultra_pos, ultra_species, ultracell_idx

