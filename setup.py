#!/usr/bin/env python
"""
setup.py <arg>

<argv> = dev sets up repository for pushing to master. For developers only.
"""

import os
import sys
import importlib

class DevUserError(Exception):
    pass

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

