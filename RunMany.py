"""
Launch many different tests with multiprocessing
"""

import multiprocessing
import subprocess

def spawnProcess(args):
    cmd = ['python','Run.py'] + [str(i) for i in args]
    subprocess.run(cmd)


#works, uses tons of memory
#for parsing the command line generally will just use the default hyper parameters
#so it is specifying the different manipulators, states, tasks, and replicates
if __name__ == "__main__":

    processes = []
    for manip in ['cable','tca']:
        for state in ['tip','both']:
            for (task,samples,replicates) in [('dynReach',50000,3), ('varTarg',200000,3), ('varTraj',1000000,1)]:
                processes = processes + [(manip,state,task,samples)]*replicates
                
    print(processes)
    
    with multiprocessing.Pool() as p:
        p.map(spawnProcess,processes)
