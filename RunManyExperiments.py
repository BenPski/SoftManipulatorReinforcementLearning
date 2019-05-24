# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 10:03:33 2018

@author: benpski


Running many of the experiments from a config file
"""


import multiprocessing
import subprocess
from configparser import ConfigParser
from argparse import ArgumentParser
import sys

def spawnProcess(args):
    cmd = ['python','RunExperiment.py'] + [str(i) for i in args]
    subprocess.run(cmd)

def processesFromConfig(config):
    #read the config to generate the processes
    #the order of the commands matters
    processes = []
    for exp in config.sections():
        params = config[exp]
        task = params['task']
        manip = params['manip']
        measure = params['measure']
        train_steps = params['train_steps']
        train_samples = params['train_samples']
        train_bound = params['train_bound']
        test_steps = params['test_steps']
        replicates = int(params['replicates'])
        if task == 'dynReach':
            train_terminal = params['train_terminal']
            test_terminal = params['test_terminal']
            commands = ('dynReach',
                        measure,
                        train_steps,
                        train_samples,
                        train_bound,
                        train_terminal,
                        test_steps,
                        test_terminal,
                        manip)
            processes += [commands]*replicates
        elif task == 'varTarg':
            train_terminal = params['train_terminal']
            test_terminal = params['test_terminal']
            test_repeat = params['test_repeat']
            commands = ('varTarg',
                        measure,
                        train_steps,
                        train_samples,
                        train_bound,
                        train_terminal,
                        test_steps,
                        test_terminal,
                        test_repeat,
                        manip)
            processes += [commands]*replicates
        elif task == 'varTraj':
            commands = ('varTraj',
                        measure,
                        train_steps,
                        train_samples,
                        train_bound,
                        test_steps,
                        manip)
            processes += [commands]*replicates
        else:
            raise Exception("Didn't recognize the task.")
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
    
    with multiprocessing.Pool(3) as p:
        p.map(spawnProcess,processes)
