"""
Module to provide interface between Python body and fortran routines.

Notes
-----
Data parsing, page reordering and general interface work is done here. 
Please don't do this in fortran!
"""

import nn_potential.fortran.assorted as assorted
import numpy as np
import copy
from parsers.atomic_data import atomic_number
#from fitelectrondensity.misc import get_ultracell
import tempfile
import shutil

#-----------#
# utilities #
#-----------#

def get_num_threads():
    """
    Return the number of threads that are used in parallelised
    sections of fortran. This is equal to the enivorment variable
    OMP_NUM_THREADS in unix. To change this, set

    export OMP_NUM_THREADS = <x> in shell before executing Python
    """
    return assorted.f90wrap_get_num_threads()


#------------------------------#
# routines for bond generation #
#------------------------------#

def query_ball_point(ultracell,eval_points,r_cut):
    atom_idxs = np.zeros((len(ultracell),len(eval_points)),dtype=np.int32,order='F')
    num_neighbours = np.zeros(len(eval_points),dtype=np.int32,order='F') 

    # need to tranpose 2d arrays for Fortran page ordering
    assorted.f90wrap_query_ball_point(ultracell.transpose(),len(ultracell),eval_points.transpose(),\
    len(eval_points),r_cut,atom_idxs,num_neighbours)

    # return same types as KDTree.query_ball_point
    return np.array([list(atom_idxs[:num_neighbours[i],i]) for i in range(len(eval_points))])

def MEAM_aniso_bonds(_el1,_el2,tmp_r,tmp_r_vec,tmp_idx_ani):
    import time
    t0 = time.time()

    Nel1 = len(tmp_idx_ani[_el1])
    Nel2 = len(tmp_idx_ani[_el2])
    same_el = _el1==_el2
    
    if same_el:
        num_pairs = int(Nel1*(Nel1-1)*0.5)
    else:
        num_pairs = Nel1*Nel2

    ani_out = np.zeros((3,num_pairs),dtype=np.float64,order='F')

    r_el1 = tmp_r[_el1][tmp_idx_ani[_el1]]
    r_el2 = tmp_r[_el2][tmp_idx_ani[_el2]]
    r_vec_el1 = np.array(tmp_r_vec[_el1][tmp_idx_ani[_el1]].transpose(),order='F')
    r_vec_el2 = np.array(tmp_r_vec[_el2][tmp_idx_ani[_el2]].transpose(),order='F')
    
    assorted.f90wrap_meam_bond_generator(r_el1,r_el2,r_vec_el1,r_vec_el2,Nel1,Nel2,\
                                         num_pairs,same_el,ani_out)
    # transpose and change page order
    return np.array(ani_out.transpose(),order='C')

def get_ultracell(fpos,cell,species,rcut,N=50):
    
    Natm = len(fpos)
    Nel = (2*N+1)**3 * Natm

    newfrac = np.zeros((3,Nel),dtype=np.float64,order='F')
   
    isneigh = np.ones(Nel,dtype=np.int32,order='F')*-1

    bondsearch.f90wrap_findneighbours(np.array(fpos.T,order='F'),Natm,\
    np.array(cell.T,order='F'),N,rcut,newfrac,isneigh)

    for i in range(len(isneigh)):
        if isneigh[i] == -1:
            break

    newfrac = np.array(newfrac.T[isneigh[:i]],order='C')

    ultra_species = [None for i in range(len(newfrac))]
    ultra_idx = [None for i in range(len(newfrac))]
    tmp_idx = list([[a[j] - np.floor(a[j]) for j in range(3)] for a in copy.deepcopy(newfrac)])
    for i in range(len(ultra_idx)):
        for j in range(len(fpos)):
            if all([np.isclose(fpos[j][k],tmp_idx[i][k]) for k in range(3)]):
                ultra_species[i] = species[j]
                ultra_idx[i] = j

    return np.dot(newfrac,cell),ultra_species,ultra_idx

#-----------------------------#
# routines for rvm regression #
#-----------------------------#

def wrapped_cos(k,r_s,r):   
    """
    interface to fortran wrapped cosine function
    """
    return assorted.f90wrap_wrapped_cos(k,r_s,r)

def wrapped_smooth(f_s,r_s,kind,r):
    """
    interface to fortran wrapped smoothing function
    """
    return assorted.f90wrap_wrapped_smooth(f_s,r_s,kind,r)


#----------------------------#
# routines for design matrix #
#----------------------------#

def isotropic_phi(info,bonds,species,multithreading=True):
    """
    Python interface to fortran wrapped fortran for generation of
    isotropic design matrix segments

    Input
    -----
        - info           : a dictionary of bond information necessary to
                           calculate the full isotropic basis set 
        - bonds          : the list of bonds to create isotropic matrix 
                           elements for
        - species        : list of atom species 
        - multithreading : if True, use OpenMP version of fortran code.
                           export OMP_NUM_THREADS=<x> in shell
                           environment before running python, to ues <x>
                           threads if present 
    """
    # phi matrix
    phi = np.zeros( (len(bonds),info["k_iso"]),dtype=np.float64,order='F')


    # this should account for the case of no 'species' neighbour in bond
    lengths = [len(bond.x["r"][species]) if species in bond.x["r"] else 0 for  bond in bonds]

    # RECENT BUG FIX
    if len(lengths)==0:
        # no contribution to phi matrix
        return np.array(phi,order='C')

    rvalues = np.zeros( (max(lengths),len(bonds)),dtype=np.float64,order='F')

    for i in range(len(bonds)):
        for j in range(lengths[i]):
            # [r][bonds]
            rvalues[j][i] = bonds[i].x["r"][species][j]
  
    
    if not info["self_contribution"]:
        # taper contribution to be zero at r=0
        taper_cutoffs = np.array([info["r_smooth"],0.0],dtype=np.float64)
    else:
        # taper_cutoofs[1] == -1 will triger no tapering at 0.0
        taper_cutoffs = np.array([info["r_smooth"],-1.0],dtype=np.float64)

    # basis function key
    basis_key = {"Fourier":0,"polynomial":1,"gaussian":2}

    if info["type_iso"] in basis_key:
        if multithreading:
            assorted.f90wrap_openmp_isotropic_phi_cos(rvalues,np.arange(info["k_iso"]),info["f_smooth"],\
                taper_cutoffs,info["smooth"],info["kspacing_iso"],info["k_iso"],len(bonds),max(lengths),\
                lengths,basis_key[info["type_iso"]],phi)
        else:
            assorted.f90wrap_isotropic_phi_cos(rvalues,np.arange(info["k_iso"]),info["f_smooth"],\
                taper_cutoffs,info["smooth"],info["kspacing_iso"],info["k_iso"],len(bonds),max(lengths),\
                lengths,basis_key[info["type_iso"]],phi)
    else:
        # need to implement s+c routine
        raise NotImplementedError

    return np.array(phi,order='C')


def anisotropic_phi(info,bonds,species,multithreading=False):
    """
    Python interface to fortran anisotropic design matrix segment
    
    Input
    -----
        - info           : a dictionary of bond information necessary to
                           calculate the full isotropic basis set 
        - bonds          : the list of bonds to create isotropic matrix 
                           elements for
        - species        : list of atom species 
        - multithreading : if True, use OpenMP version of fortran code.
                           export OMP_NUM_THREADS=<x> in shell
                           environment before running python, to ues <x>
                           threads if present 
    """
    import itertools
    import time


    if species[0]==species[1]:
        same_species = True
    else:
        same_species = False
    
    # create k values array [1:3,:]
    if same_species:
        kvals_tmp  = list(itertools.product(range(info["k_ani"]["like"]["r"]),range(info["k_ani"]["like"]["theta"]))) 
        kvals = np.array([[k[0],k[0],k[1]] for k in kvals_tmp],dtype=np.int32,order='F').T
    else:
        kvals = np.array(list(itertools.product(range(info["k_ani"]["unlike"]["r"]),range(info["k_ani"]["unlike"]["r"]),range(info["k_ani"]["unlike"]["theta"]))),dtype=np.int32,order='F').T
    
    Nk = np.shape(kvals)[1]
    
    # phi matrix
    phi = np.zeros( (len(bonds),Nk),dtype=np.float64,order='F')

    # number of neighbours in each bond
    lengths = [len(bond.x["ani"][species]) if species in bond.x["ani"] else 0 for bond in bonds]

    if len(lengths)==0:
        # no contribution to phi matrix
        return np.array(phi,order='C')

    r_thetas = np.zeros( (len(bonds),max(lengths),3),dtype=np.float64,order='C')
    for i in range(len(bonds)):
        if species not in bonds[i].x["ani"]:
            continue
        r_thetas[i][0:lengths[i]][:] = bonds[i].x["ani"][species][:][:]
   
    # [property][neighbour in bond][bond]
    r_thetas = np.array(r_thetas.transpose(),order='F')

    # iso and aniso smoothing scaling
    smoothing = np.array([info["f_smooth"],info["ani_specification"]["f_smooth"]],dtype=np.float64)

    # key for basis set type in fortran
    if info["ani_type"] == "MEAM":
        function_type = 0
    elif info["ani_type"] == "polynomial":
        function_type = 1
    else:
        raise NotImplementedError

    print ('doing aniso design matrix for function type {} {} '.format(info["ani_type"],function_type))
    t1 = time.time()
    if info["type_ani"] == "Fourier":
        if multithreading:
            assorted.f90wrap_openmp_anisotropic_phi_cos(r_thetas,kvals,smoothing,\
                info["ani_specification"]["r_ani"],info["smooth"],np.array([info["ani_specification"]["kspacing"]["distance"],info["ani_specificaion"]["kspacing"]["angle"]],dtype=np.float64),\
                Nk,len(bonds),max(lengths),lengths,\
                function_type,phi)
        else:
            assorted.f90wrap_anisotropic_phi_cos(r_thetas,kvals,smoothing,\
                info["ani_specification"]["r_ani"],info["smooth"],np.array([info["ani_specification"]["kspacing"]["distance"],info["ani_specificaion"]["kspacing"]["angle"]],dtype=np.float64),\
                Nk,len(bonds),max(lengths),lengths,\
                function_type,phi)
    else:
        raise NotImplementedError
    t2 = time.time()
    print ('fortran part for aniso: {}'.format(t2-t1))
    
    return np.array(phi,order='C')

def phi_singlestructuref90(s,rcut,basis_size,k=1.0):
    """
    wrapper to fortran to return design matrix of a single structure
    """

    # nearest image calculation not yet support in fortran
    #ultra_pos,ultra_species,_ = fed.misc.get_ultracell(fpos=s["positions"],cell=s["cell"],\
    #        species=s["species"],r_cut=rcut)

    # need to convert fractional to cartesians

    Zultra = np.asarray([atomic_number.atomic_number[_s] for _s in ultra_species],dtype=np.int32)

    # ultra atoms are in same order as local atoms. Index from 1.
    ultraidx = []
    while len(ultraidx)<len(ultra_pos):
        ultraidx += range(1,len(s["positions"])+1)
    ultraidx = np.asarray(ultraidx,dtype=np.int32)
    print (ultraidx)

    phi = np.zeros((basis_size["M1"]*basis_size["M2"]+basis_size["M3"]*basis_size["M4"],len(s["positions"])),\
            dtype=np.float64,order='F')

    assorted.f90wrap_design_matrix_singlestructure(rcut=rcut,\
            localpos=np.array(np.dot(s["positions"],s["cell"]).T,order='F',dtype=np.float64),\
            ultrapos=np.array(ultra_pos.T,order='F',dtype=np.float64),\
            zlocal=np.array(s["atomic_number"],dtype=np.float64),\
            zultra=Zultra,natm=np.array([len(s["species"]),len(Zultra)],dtype=np.int32),\
            invk2=1.0/(k**2),ultraidx=ultraidx,\
            basis=np.asarray([basis_size["M1"],basis_size["M2"],basis_size["M3"],basis_size["M4"]],\
            dtype=np.int32),\
            phi=phi)

    return np.asarray(phi.T,order='C')           


#-------------------------------#
# routines for sedc calculation #
#-------------------------------#

def write_f90inputfile(fname,structure):
    """
    1 | ex  ey  ez
    2 | l1x l1y l1z
    3 | l2x l2y l2z
    4 | l3x l3y l3z
    5 |
    6 | f1 f2 f3 Z
    7 | .. .. .. ..
    8 | .. .. .. ..
    . | .. .. .. ..
    . | EOF
    
    """
    with open(fname,'w') as f:
        f.write('{:<20} {:<20} {:<20}\n'.format('ex','ey','ez'))
        for ii in range(3):
            f.write('{:<20} {:<20} {:<20}\n'.format(structure["cell"][ii][0],\
                    structure["cell"][ii][1],structure["cell"][ii][2]))
        # need to write non-escape character in line for f90 read to work as assumed
        f.write('{:<64}\n{:<20} {:<20} {:<20} {:<20}\n'.format(''.join(['-' for ii in range(64)]),\
                'f1','f2','f3','Z'))
        for ii,_pos in enumerate(structure["positions"]):
            f.write('{:<20} {:<20} {:<20} {:<20}\n'.format(_pos[0],_pos[1],_pos[2],\
                    structure["atomic_number"][ii]))


def design_matrixf90(gips,rcuts,gammaks,Zks,basis,phi_type,multithreading=True):
    assert isinstance(rcuts,(list,np.ndarray)),"wrong type"
    assert isinstance(gammaks,(list,np.ndarray)),"wrong type"
    assert isinstance(Zks,(list,np.ndarray)),"wrong type"
    
    if phi_type==1:
        # number of basis functions
        Nbases = basis["M1"]*basis["M2"]+basis["M3"]*basis["M4"]
    elif phi_type == 2:
        Nbases = basis["M1"]*basis["M2"]*basis["M3"]*basis["M4"]

    # need to write to disk for i/o
    tmpdir = tempfile.mkdtemp(prefix='potty-')
    
    nfiles,files = [],[]
    
    for _gip in gips:
        for _s in _gip:
            _fname = tmpdir+'/sedc_{}.in'.format(len(nfiles)+1)
            
            write_f90inputfile(fname=_fname,structure=_s)
            
            nfiles.append(len(_s["positions"]))
            files.append('{:<1024}'.format(_fname))
   
    phi = np.zeros((Nbases,np.sum(np.array(nfiles))),order='F',\
        dtype=np.float64)

    # Please Don't change this!
    chararray = np.array(files,dtype='c').T

    # number of basis functions
    Nbases = basis["M1"]*basis["M2"]+basis["M3"]*basis["M4"]

    if multithreading:
        assorted.f90wrap_phi_multiple_structures_openmp(len(nfiles),chararray,np.array(nfiles),\
                rcuts,gammaks,Zks,np.array([basis["M1"],basis["M2"],basis["M3"],basis["M4"]],dtype=np.int32),\
                Nbases,phi_type,phi)
    else:
        assorted.f90wrap_phi_multiple_structures(len(nfiles),chararray,np.array(nfiles),\
                rcuts,gammaks,Zks,np.array([basis["M1"],basis["M2"],basis["M3"],basis["M4"]],dtype=np.int32),\
                Nbases,phi_type,phi)
   
    # finished with scratch files
    shutil.rmtree(tmpdir)

    # Want Python user to see C page ordering
    return np.array(phi.T,order='C')


def gamma_featuref90(gips,rcut,gammak=None,Zk=None,precision=None,mean=None,normal_zk=None):
    if gammak!=None and Zk!=None:
        parameters = np.asarray([gammak,Zk],dtype=np.float64)
    elif precision is not None and mean is not None:
        parameters = np.asarray([mean,precision,normal_zk[0],normal_zk[1]],dtype=np.float64)

    # need to write to disk for i/o
    tmpdir = tempfile.mkdtemp(prefix='potty-')
    
    nfiles,files = [],[]
    
    for _gip in gips:
        for _s in _gip:
            _fname = tmpdir+'/sedc_{}.in'.format(len(nfiles)+1)
            
            write_f90inputfile(fname=_fname,structure=_s)
            
            nfiles.append(len(_s["positions"]))
            files.append('{:<1024}'.format(_fname))

    gamma_features = np.zeros(np.sum(np.array(nfiles)),dtype=np.float64)


    # Please Don't change this!
    chararray = np.array(files,dtype='c').T
    
    assorted.f90wrap_gamma_multiple_structures_openmp(len(nfiles),chararray,np.array(nfiles),\
        rcut,parameters,parameters.shape[0],gamma_features)
   
    # finished with scratch files
    shutil.rmtree(tmpdir)

    # Want Python user to see C page ordering
    return gamma_features

def angular_infof90(gips,rcut):

    # need to write to disk for i/o
    tmpdir = tempfile.mkdtemp(prefix='potty-')
    
    nfiles,files = [],[]
    
    for _gip in gips:
        for _s in _gip:
            _fname = tmpdir+'/sedc_{}.in'.format(len(nfiles)+1)
            
            write_f90inputfile(fname=_fname,structure=_s)
            
            nfiles.append(len(_s["positions"]))
            files.append('{:<1024}'.format(_fname))


    # Please Don't change this!
    chararray = np.array(files,dtype='c').T
    
    # fetch size of array to allocate on Python stack
    num_triplets = assorted.f90wrap_threebody_angular_info( len(nfiles),chararray,np.array(nfiles),\
        rcut,True,1,np.zeros((3,1),dtype=np.float64,order='F') )
 
    angular_info = np.zeros((3,num_triplets),dtype=np.float64,order='F')


    _ = assorted.f90wrap_threebody_angular_info( len(nfiles),chararray,np.array(nfiles),\
        rcut,False,num_triplets,angular_info )

    # finished with scratch files
    shutil.rmtree(tmpdir)

    # Want Python user to see C page ordering
    return np.asarray(angular_info.T,order='C')

def anisotropic_featuref90(gips,rcut,means,precisions,normal_zk):

    # need to write to disk for i/o
    tmpdir = tempfile.mkdtemp(prefix='potty-')
    
    nfiles,files = [],[]
    
    for _gip in gips:
        for _s in _gip:
            _fname = tmpdir+'/sedc_{}.in'.format(len(nfiles)+1)
            
            write_f90inputfile(fname=_fname,structure=_s)
            
            nfiles.append(len(_s["positions"]))
            files.append('{:<1024}'.format(_fname))

    num_components = precisions.shape[0]

    aniso_features = np.zeros((num_components,np.sum(np.array(nfiles))),dtype=np.float64,order='F')

    # Please Don't change this!
    chararray = np.array(files,dtype='c').T
    
    assorted.f90wrap_anisotropic_feature_multiple_structures_openmp(len(nfiles),chararray,np.array(nfiles),\
        rcut,num_components,np.asarray(means.T,order='F'),np.asarray(precisions.T,order='F'),\
        normal_zk[0],normal_zk[1],aniso_features)
   
    # finished with scratch files
    shutil.rmtree(tmpdir)

    # Want Python user to see C page ordering
    return np.asarray(aniso_features.T,order='C')

def gamma_feature(dr2_list,Z,gammak,Zk,rcut):
    return assorted.f90wrap_gamma_gaussian(np.asarray(dr2_list,dtype=np.float64),\
            np.asarray(Z,dtype=np.float64),len(dr2_list),1.0/(gammak**2),Zk,rcut)

def atomatomdistances(gips,rcut):

    # need to write to disk for i/o
    tmpdir = tempfile.mkdtemp(prefix='potty-')
    
    nfiles,files = [],[]
    
    for _gip in gips:
        for _s in _gip:
            _fname = tmpdir+'/sedc_{}.in'.format(len(nfiles)+1)
            
            write_f90inputfile(fname=_fname,structure=_s)
            
            nfiles.append(len(_s["positions"]))
            files.append('{:<1024}'.format(_fname))


    # Please Don't change this!
    chararray = np.array(files,dtype='c').T

    # count number of atom atom distances among all structures, within rcut
    Ndistances = assorted.f90wrap_num_atom_atom_distances(len(nfiles),chararray,np.array(nfiles),rcut)

    print('have found {} distances'.format(Ndistances))

    distances = np.zeros(Ndistances,dtype=np.float64)

    assorted.f90wrap_atom_atom_distances(len(nfiles),chararray,np.array(nfiles),rcut,Ndistances,distances)

    # finished with scratch files
    shutil.rmtree(tmpdir)

    # Want Python user to see C page ordering
    return distances
