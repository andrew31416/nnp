#!/usr/bin/env python
"""
setup.py <arg>

<argv> = dev sets up repository for pushing to master. For developers only.
"""

import os
import sys
import importlib
import subprocess
import warnings

class DevUserError(Exception):
    pass

def GeneralMessage(message):
    print('{}\n{}\n{}\n\n{}'.format(''.join(['*' for ii in range(36)]),'Setup issue with current environment',\
            ''.join(['*' for ii in range(36)]),message))    
    sys.exit()

class fortran_checks():
    def __init__(self):
        self.compile_dir = "./nn/fortran"
        self.fortran_api = "nn_f95"

        if self.check_compiler() and self.check_f2py():
            # check for linear algebra libraries
            self.check_libraries()
            
            # OK compiler found
            self.compile_fortran() 

    def check_compiler(self):
        """ 
        need gfortran 5.x.y or higher
        """
        try:
            process = subprocess.run(["gfortran","--version"],stdout=subprocess.PIPE)
            version = process.stdout.decode("utf-8").split('\n')[0].split()[-2]
        
            if int(version.split('.')[0]) <= 4:
                GeneralMessage("Version of gfortran found in current path list is {}, need at least version 5 (recommend 5.4.0)".\
                        format(version))
            success = True
        except FileNotFoundError:
            GeneralMessage("Cannot find gfortran, please install or ensure is in current path list before reunning setup.py")   
        return success          

    def check_f2py(self):
        """
        Check for f2py and f90wrap
        """
        recommended_version = {"f2py":"2","f90wrap":"0.1.4"}
        version_cmd = {"f2py":"-v","f90wrap":"--version"}

        for _exec in ["f2py","f90wrap"]:
            try:
                process = subprocess.run([_exec,version_cmd[_exec]],stdout=subprocess.PIPE)
                version = process.stdout.decode("utf-8").split()[-1].strip("v")
            
                if version != recommended_version[_exec]:
                    warnings.warn("{} version found is {}, recommend version {}".format(_exec,version,recommended_version[_exec]))
            
            except FileNotFoundError:
                GeneralMessage("Cannot find {} in pathlist, please install or ensure is in current path list.")
        return True

    def compile_fortran(self):
        """
        Compile fortran
        """
        os.chdir(self.compile_dir)                

        # attempt to compile
        process = subprocess.run(["./build.sh"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        
        # import_module relies on imported modules being in sys.path
        sys.path.append(os.getcwd())        
        
        try:
            # attempt import of fortran 
            importlib.import_module(self.fortran_api)
        except ModuleNotFoundError:
            GeneralMessage("Compilation of fortran API has failed. Move to nn/fortran and run ./build.sh for verbose output")

    def check_libraries(self):
        """
        check lapack and blas are accessible with gfortran compiler
        """
        flines = ['program main\nreal(8),external :: ddot\nreal(8) :: x(1:2),y(1:2)\nreal(8) :: z\nz=ddot(2,x,1,y,1)\nend program']
        scratch_file = 'scratch.f90'        
        with open(scratch_file,'w') as f:
            f.writelines(flines)

        # compile toy main, attempting to link to lapack and blas
        process = subprocess.run(["gfortran","scratch.f90","-lblas","-llapack"],stdout=subprocess.PIPE,\
                stderr=subprocess.PIPE)
        stderr = process.stderr.decode('utf-8')
    
        # tidy up scratch files
        remove_me = [_f for _f in ["a.out",scratch_file] if os.path.exists(_f)]        
        for _f in remove_me:
            os.remove(_f)
        
        missing_libs = []
        for _l in stderr.split('\n'):
            if 'cannot find -l' in _l:
                missing_libs.append(_l.split()[-1])
        if len(missing_libs)>0:
            GeneralMessage("Fortran makes extensive use of linear algebra libraries. Cannot find:\n\n{}\n\n{}".format(\
                    "\n".join(missing_libs),'Please make these accessible to linker before rerunning setup.py'))
         
 
if __name__ == "__main__":
    supported_args = ["dev"]

    if len(sys.argv)>1:
        if sys.argv[1] == "dev":
            # hook to include with standard git commands
            hooks = ['pre-push']

            for _hook in hooks:
                # add to git hooks directory
                os.symlink(src='../../setup/{}'.format(_hook),dst='.git/hooks/{}'.format(_hook))
        else:
            print("Unrecognised argument {} passed to setup.py. Did you mean one of {} ?".format(\
                    sys.argv[1],", ".join(supported_args)))
            sys.exit(0)

    # check for required modules in ./setup/required_modules.txt
    if not os.path.exists('./setup/required_modules.txt'):
        raise DevUserError("required_modules.txt file not present, run ./setup/generate_used_modules.py")
    else:
        with open('./setup/required_modules.txt','r') as f:
            modules = [_l.strip('\n') for _l in f.readlines()]
        
            missing_modules = []                
            for _mod in modules:
                try:
                    importlib.import_module(_mod)
                except ModuleNotFoundError:
                    missing_modules.append(_mod)
            
            if len(missing_modules)!=0:
                GeneralMessage('The following modules cannot be found:\n\n{}\n\nPlease install before rerunning setup.py'.format(\
                        '\n'.join(missing_modules)))


    # check for compatability of fortran components
    check = fortran_checks()
