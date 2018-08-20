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
    print(message)
    sys.exit()

class fortran_checks():
    def __init__(self):
        self.compile_dir = "./nn/fortran"

        if self.check_compiler() and self.check_f2py():
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
        process = subprocess.run(["./build.sh"])
 
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
                print('The following modules cannot be found:\n\n')
                print('\n'.join(missing_modules))
                print('\n\nPlease install before rerunning setup.py')


    # check for compatability of fortran components
    check = fortran_checks()
