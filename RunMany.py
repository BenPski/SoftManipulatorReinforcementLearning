"""
Launch many different tests with multiprocessing
"""

import multiprocessing
import subprocess
import argparse

def spawnProcess(args):
    cmd = ['python','Run.py'] + [str(i) for i in args]
    subprocess.run(cmd)


#works, uses tons of memory
#for parsing the command line generally will just use the default hyper parameters
#so it is specifying the different manipulators, states, tasks, and replicates
if __name__ == "__main__":

    processes = [('cable','tip','dynReach',10000)]*3
    with multiprocessing.Pool() as p:
        p.map(spawnProcess,processes)

    #p = multiprocessing.Process(target = run, args=('cable','tip','dynReach',1000))
    #p.start()
    #p.join()
