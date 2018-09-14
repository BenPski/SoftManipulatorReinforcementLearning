# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:22:43 2018

@author: benpski

Output the relevant data for displaying the different training tasks.

"""

from argparse import ArgumentParser
from config_utils import ConfigHandler
import os
import matplotlib.pyplot as plt

def collectPerfs(name):
    cable_tip = []
    cable_both = []
    tca_tip = []
    tca_both = []
    
    for f in os.listdir('./tests/'):
        if name in f:
            perf = ConfigHandler.fromFile('./tests/'+f).perf
            if 'cable_tip' in f:
                cable_tip.append(perf)
            elif 'cable_both' in f:
                cable_both.append(perf)
            elif 'tca_tip' in f:
                tca_tip.append(perf)
            elif 'tca_both' in f:
                tca_both.append(perf)
            else:
                print("Something bonkers happened")
    return cable_tip, cable_both, tca_tip, tca_both

def dynReachPerfs():
    """
    collect the successful reaches for the dynamic reaching experiments
    want to show
        cable, tip: n/N
        cable, both: n/N
        tca, tip: n/N
        tca, both: n/N
    """
    cable_tip, cable_both, tca_tip, tca_both = collectPerfs('dynReach')
            
    shower = lambda xs: str(sum(xs))+"/"+str(len(xs))
    print("Results of Dynamic Reaching Learning:")
    print("Cable & Tip: ", shower(cable_tip))
    print("Cable & Both: ", shower(cable_both))
    print("TCA & Tip: ", shower(tca_tip))
    print("TCA & Both: ", shower(tca_both))
    print()
    
def varTargPerfs():
    """
    want to show:
        cable, tip: n/100, p%
        cable, both: n/100, p%
        tca, tip: n/100, p%
        tca, both: n/100, p%
    """
    cable_tip, cable_both, tca_tip, tca_both = collectPerfs('varTarg')
            
    mean = lambda xs: sum(xs)/len(xs)
    shower = lambda xs: str(mean(list([i[0] for i in xs]))) + ", " + str(100*mean(list(i[1] for i in xs))) + "%" + " | Best " + str(max(xs)) + " | Worst " + str(min(xs))
    
    print("Results of Variable Target Learning:")
    print("Cable & Tip: ", shower(cable_tip))
    print("Cable & Both: ", shower(cable_both))
    print("TCA & Tip: ", shower(tca_tip))
    print("TCA & Both: ", shower(tca_both))
    print()
    
def varTrajPerfs():
    """
    want to plot the average tip errors
    want the plots seperated into cable and tca and to have those individually plot tip and both 
    """
    
    cable_tip, cable_both, tca_tip, tca_both = collectPerfs('varTraj')
    
    mean = lambda xs : sum(xs)/len(xs)
    mean_err = lambda xs: [mean([i[0] for i in xs]) for j in range(len(xs[0]))]
    scale = lambda s, xs: [s*i for i in xs]
    tip_errors = lambda xs: scale(100,mean_err(xs))
    
    if len(cable_tip) > 0 and len(cable_both) > 0:
        cable_tip = tip_errors(cable_tip)
        cable_both = tip_errors(cable_both)
    
        fig, ax = plt.subplots()
        ax.plot(cable_tip, label="Tip")
        ax.plot(cable_both, label="Tip and Middle")
        ax.legend()
        plt.savefig("./perfs/cable_trajectory.png")
        print("Cables figure saved to ./perfs/cable_trajectory.png")
        
    if len(tca_tip) > 0 and len(tca_both) > 0:
        tca_tip = tip_errors(tca_tip)
        tca_both = tip_errors(tca_both)
    
        fig, ax = plt.subplots()
        ax.plot(tca_tip, label="Tip")
        ax.plot(tca_both, label="Tip and Middle")
        ax.legend()
        plt.savefig("./perfs/tca_trajectory.png")
        print("TCAs figure saved to ./perfs/tca_trajectory.png")
                
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Collect together the relevant data for displaying the resulting performances.")
    
    #want to be able to select specific types of experiments or just do all of them as the default
    parser.add_argument('--dynReach', action='store_true', help='Show performances of Dynamic Reaching experiments.')
    parser.add_argument('--varTarg', action='store_true', help='Show performances of Variable Targets experiments.')
    parser.add_argument('--varTraj', action='store_true', help='Show performances of Variable Trajectories experiments.')
    
    args = parser.parse_args()
    
    if not args.dynReach and not args.varTraj and not args.varTarg:
        dynReachPerfs()
        varTargPerfs()
        varTrajPerfs()
    else:
        if args.dynReach:
            dynReachPerfs()
        if args.varTarg:
            varTargPerfs()
        if args.varTraj:
            varTrajPerfs()
    