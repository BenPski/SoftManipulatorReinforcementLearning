"""
Launch many different tests with multiprocessing
"""

import multiprocessing
import subprocess
from configparser import ConfigParser
from argparse import ArgumentParser
import sys

def spawnProcess(args):
    cmd = ['python','Run.py'] + [str(i) for i in args]
    subprocess.run(cmd)

def processesFromConfig(config):
    #read the config to generate the processes
    processes = []
    for exp in config.sections():
        params = config[exp]
        manip = params['manip']
        state = params['state']
        task = params['task']
        samples = int(params['samples'])
        steps = int(params['steps'])
        replicates = int(params['replicates'])
        processes += [(manip,state,task,samples,steps)]*replicates
    return processes

if __name__ == "__main__":

    parser = ArgumentParser(description="Run many learning processes at once. Provide a config that specifies the processes to run.")
    parser.add_argument('config', type=str, help="The location of the config to read that specifies the processes.")

    args = parser.parse_args()

    config = ConfigParser()
    try:
        with open(args.config,'r') as f:
            config.read_file(f)
    except:
        sys.exit("Unable to read specified config.")

    processes = processesFromConfig(config)


    #processes = []
    #for manip in ['cable','tca']:
    #    for state in ['tip','both']:
    #        for (task,samples,replicates) in [('dynReach',50000,3), ('varTarg',200000,3), ('varTraj',1000000,1)]:
    #            processes = processes + [(manip,state,task,samples)]*replicates

    print(processes)

    with multiprocessing.Pool(4) as p:
        p.map(spawnProcess,processes)
