#!/usr/bin/env python
"""
setup.py <arg>

<argv> = dev sets up repository for pushing to master. For developers only.
"""

import os
import sys


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
            print("Unrecognised argument {} passed to setup.py. Did you mean one of {} ?".format(sys.argv[1],", ".join(supported_args)))
            sys.exit(0)

