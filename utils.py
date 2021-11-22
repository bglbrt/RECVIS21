#!/usr/bin/env python

# os libraries
import os

def ignore_files(dir, files):
    '''
    Arguments:
        dir: str
            - path to directory
        files:
            - list of paths to files

    Returns:
        _: list
            - list of files to ignore
    '''
    return set(f for f in files if os.path.isfile(os.path.join(dir, f)))
