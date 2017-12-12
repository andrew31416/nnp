import numpy as np
import tempfile
import shutil
import nnp.nn.fortran.nn_f95 as f95_api

def _write_file(config,fname):
    """_write_file

    Write a single configuration to disk for parsing from fortran
    
    1 | ex   ey  ez
    2 | l1x l1y l1z
    3 | l2x l2y l2z
    4 | l3x l3y l3z
    5 | -----------
    6 | rx ry rz Z  fx fy fz
    7 | .. .. .. .. .. .. ..
    . | .. .. .. .. .. .. ..
    . | .. .. .. .. .. .. ..
    . | ----------
    . | E_tot
    . | EOF
    
    Parameters
    ----------
    config : parsers.supercell
        A single atomic configuration using the parsers module data structure

    fname :  String
        Path (relative or full) to new file destination
    
    Examples
    --------
    >>> import parsers
    >>> import nnp
    >>> config = parsers.supercell()
    >>> config["cell"] = np.asarray([[5.0,0.0,0.0],[0.0,5.0,0.0],[0.0,0.0,5.0]])
    >>> config["positions"] = np.asarray([[0.0,0.0,0.0],[0.0,0.5,0.5]])
    >>> config["atomic_number"] = np.asarray([4,6])
    >>> nnp.util.io._write_file(config,'atoms.config')
    """

    for _attr in ["cell","positions","atomic_number"]:
        if getattr(config,_attr) is None:
            raise IoError("attribute {} is None but must be set before writing".format(_attr))
    
    _val = {"forces":None,"energy":None}
    
    for _attr in ["forces","energy"]:
        # if forces or energy not present, write as 0
        if getattr(config,_attr) is None:
            if _attr == "forces":
                _val[_attr] = np.zeros((getattr(config,"atomic_number").shape[0],3),dtype=np.float64)
            else:
                _val[_attr] = 0.0
        else:
            _val[_attr] = getattr(config,_attr)

    with open(fname,'w') as f:
        f.write('{:<20} {:<20} {:<20}\n'.format('ex','ey','ez'))
        
        # cell vectors
        for ii in range(3):
            f.write('{:<20} {:<20} {:<20}\n'.format(config["cell"][ii][0],\
                    config["cell"][ii][1],config["cell"][ii][2]))
        # need to write non-escape character in line for f90 read to work as assumed 
        f.write('{:<145}\n{:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20} \n'.\
                format(''.join(['-' for ii in range(145)]),\
                'rx','ry','rx','Z','fx','fy','fz'))

        # cartesian positions, atomic number, forces
        for ii,_pos in enumerate(config["positions"]):
            _r = np.dot(config["cell"].T,_pos)
            f.write('{:<20} {:<20} {:<20} {:<20} {:<20} {:<20} {:<20}\n'.\
                    format(_r[0],_r[1],_r[2],config["atomic_number"][ii],_val["forces"][ii,0],\
                    _val["forces"][ii,1],_val["forces"][ii,2]))
       
        # total energy 
        f.write('{:<145}\n{:<20}'.format(''.join(['-' for ii in range(145)]),_val["energy"]))


def _parse_configs_to_fortran(gip,set_type):
    """parse_configs_to_fortran

    Write all structures to disk and then read into a given set type in fortran
    
    Parameters
    ----------
    gip : parsers.GeneralInputParser
        Datastructure holding a number of configurations

    set_type : String, allowed values = ['test','train']
        Position in fortran data structure to allocate to this data set
    
    Examples
    --------
    >>> import parsers
    >>> import nnp
    >>> gip = parsers.GeneralInputParser()
    >>> gip.parse_all('.')
    >>> nnp.util.io._parse_configs_to_fortran(gip,"test")
    """

    if set_type not in ['test','train']:
        raise IoError('{} not a supported set type : {}'.format(set_type,'test,train'))

    tmpdir = tempfile.mkdtemp(prefix='nnp-')
    
    _files = [] 
    for ii,_s in enumerate(gip):
        _fname = tmpdir+'/conf_{}.in'.format(ii+1) 
        _files.append('{:<1024}'.format(_fname))
       
        try: 
            _write_file(config=_s,fname=_fname)
        except IoError:
            shutil.rmtree(tmpdir) 
            raise IoError("_write_file has aborted")
   
    # fortran is fussy about type casting 
    files = np.array(_files,dtype='c').T


    _map = {"train":1,"test":2}
    # parse all text files into fortran data structures
    getattr(f95_api,"f90wrap_init_configs_from_disk")(files,_map[set_type]) 

    shutil.rmtree(tmpdir) 


class IoError(Exception):
    pass            