#!/usr/bin/env python
"""
generate_used_modules.py

Generate a plain text file required_modules.txt containing every module 
required by this package

Return Value
------------
res - int : 0,1
    If list of modules has been updated, return 1. Otherwise return 0        
"""

from modulefinder import ModuleFinder
import os
import sys

def parse_module(module):
    if len(module.split('.'))>0:
        module = module.split('.')[0]
    return module.strip('\n')

def find_modules(fname):
    modules = []
    with open(fname,'r') as f:
        flines = [_l for _l in f.readlines() if len(_l.split())>0]

        # lines where import statements are made
        idx = [ii for ii,_l in enumerate(flines) if _l.split()[0] in ["from","import"]]

        # subset of lines corresponding to module imports
        flines = [flines[_id] for _id in idx]

        for _line in flines:
            if _line[:7] == "import " and ", " not in _line:
                if " as" in _line:
                    modules.append(parse_module(_line[7:_line.find(" as ")]))
                else:
                    modules.append(parse_module(_line[7:]))
            elif _line[:5] == "from ":
                modules.append(parse_module(_line[5:_line.find(" import ")]))
            elif ", " in _line:
                _line = _line[7:].split(", ")
                modules = modules+parse_module(_line)
    return modules    
    
if __name__ == "__main__":
    # list for all imported modules
    required_modules = []

    for _root,_,_files in os.walk('..'):
        for _f in _files:
            if len(_f.split('.'))>0:
                if _f.split('.')[-1] != 'py':
                    continue
            # iterate over every .py file in subdir from module root
            required_modules += find_modules('{}/{}'.format(_root,_f))
    required_modules = list(set(required_modules))    

    # text file containing required modules
    fname = "required_modules.txt"

    # see if need to rewrite text file containing required modules
    update_file = False
    if os.path.exists(fname):
        write_new = False
        with open(fname,'r') as f:
            flines = [_line.strip('\n') for _line in f.readlines()]
        
            if set(required_modules)!=set(flines):
                # there are differences between old and current modules
                update_file = True
    else:
        # no current file
        update_file = True
        write_new = True
    
    if update_file:
        flines = '\n'.join(list(set(required_modules)))
        with open('required_modules.txt','w') as f:
            f.writelines(flines)

    # return 1 for update in file, 0 otherwise
    sys.exit({True:1,False:0}[update_file])
