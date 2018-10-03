# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:08:57 2018

@author: benpski


Just collect all the performances

"""

import subprocess
import multiprocessing
import os

def test_process(f):
    cmd = ['python','Test.py', f]
    print(cmd)
    subprocess.run(cmd)

if __name__ == "__main__":
    files = []
    for manip in ['tca']:
        for state in ['tip','both']:
            for task in ['varTarg']:
                name = manip+'_'+state+'_'+task
                files = files + ['./tests/'+i for i in os.listdir('tests/') if name in i]
                
    with multiprocessing.Pool(4) as p:
        p.map(test_process, files)