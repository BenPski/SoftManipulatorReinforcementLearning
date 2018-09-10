"""
recording the performance of a learned policy

read in a config and run the relevant tests on it
"""

import EnvironmentWrappers as EW
from manipAndWorkspace import manips, workspaces
from network_utils import generateAgent

import argparse
import configparser
import datetime
import uuid

import os
import sys
import pickle

import numpy as np

class Tester(object):
    """
    Tests the policy by reading the parameters from a config
    """

    def __init__(self,config_location):

        self.config_location = config_location
        self.config = configparser.ConfigParser()
        if os.path.isfile(config_location):
            self.config.read(config_location)
        else:
            sys.exit("The given config does not exist.")

        self.manip_name = self.config['Manipulator']['manip']
        self.manip = manips[self.manip_name]
        self.static_workspace = workspaces[self.manip_name+'_static']
        self.dynamic_workspace = workspaces[self.manip_name+'_dynamic']

        self.state = self.config['State']['state']
        self.tip = self.state=='tip'

        self.task = self.config['Task']['task']
        if self.task == 'dynReach':
            self.target = self.config['Task']['target']
            self.target = np.array(list(map(float,self.target.strip('[').strip(']').split())))
            print(self.target)
            self.env = EW.DynamicReaching(self.manip, self.target, tipOnly = self.tip)
        elif self.task == 'varTarg':
            self.env = EW.VariableTarget(self.manip, self.dynamic_workspace, tipOnly = tip)
        elif self.task == 'varTraj':
            self.tau = float(self.config['Task']['tau'])
            self.env = EW.VariableTrajectory(self.manip, self.static_workspace, self.tau, tipOnly = tip)


        self.actor_hidden = self.config['Networks']['actor_hidden']
        self.actor_hidden = list(map(int,self.actor_hidden.strip('[').strip(']').split(','))) #bit of a hacky way to be parsing the lists
        self.actor_act = self.config['Networks']['actor_act']
        self.critic_hidden = self.config['Networks']['critic_hidden']
        self.critic_hidden = list(map(int,self.critic_hidden.strip('[').strip(']').split(',')))
        self.critic_act = self.config['Networks']['critic_act']

        self.agent = generateAgent(self.env,self.actor_hidden,self.actor_act,self.critic_hidden,self.critic_act)
        self.agent.load_weights(self.config['Networks']['weights_location'])


        self.perf = None


    def run(self):
        #run the tests

        if self.task == 'dynReach':

            #for dynamic reaching just want to see if the target is reached
            obs = self.env.reset()
            reached = False
            for _ in range(100):
                a = self.agent.forward(obs)
                obs, reward, terminal, _ = self.env.step(a)
                if terminal:
                    reached = True
                    break

            self.perf = reached

        elif self.task == 'varTarg':

            #for variable target want to try several points to see how it performs

            reaches = 0
            errs = []
            for sample in range(100):
                target = self.dynamic_workspace.sample()
                reached = False
                obs = self.env.reset()
                obs[:3] = self.target
                self.env.setGoal(self.target)
                for _ in range(100):
                    a = self.agent.forward(obs)
                    obs, reward, terminal , _ = self.env.step(a)
                    if terminal:
                        reaches += 1
                        reached = True
                        break
                if not reached:
                    pos_err = np.linalg.norm(obs[:3]-obs[6:9]) #the last tip error
                    errs.append(pos_err/0.04) #normalized by length

            self.perf = (reaches,np.mean(errs))

        elif self.task == 'varTraj':

            #want to see how well a circle can be tracked

            z,r = selectTrajectoryParams(self.static_workspace)
            steps = 500
            w = 2*np.pi/steps
            circle = lambda t: np.array([r*np.cos(w*t),r*np.sin(w*t),z])
            circle_vel = lambda t: np.array([-r*w*np.sin(w*t),w*r*np.cos(w*t),0])
            self.env.manualGoal = True
            obs = self.env.reset()
            errs = []
            for t in range(steps):
                pos = circle(t)
                vel = circle_vel(t)
                goal = np.concatenate([pos,vel])
                self.env.setGoal(goal)
                obs[:6] = goal

                a = self.agent.forward(obs)
                obs, _ ,_ , _ = env.step(a)
                #env.render()
                errs.append(np.linalg.norm(goal[:3]-obs[6:9])/0.04)

            self.perf = errs

        self.save()

    def save(self):
        #save the performance measures to the perf directory
        #also update config to say that it was tested and where the file is

        try:
            os.makedirs('perfs')
        except:
            pass

        #pickle the performance measure and save to perfs

        name = "./perfs/"+self.manip_name+'_'+self.state+'_'+self.task+'_'+str(uuid.uuid4())+'.pkl'
        with open(name,'wb') as f:
            pickle.dump(self.perf, f)
        self.config['Test'] = {'perf_location': name}

        with open(self.config_location,'w') as f:
            self.config.write(f)

        print("Config updated with performance data and location.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing a learned policy")
    parser.add_argument('config', metavar='config',type=str,help='The config that stores a tests data')

    args = parser.parse_args()

    test = Tester(args.config)

    test.run()
