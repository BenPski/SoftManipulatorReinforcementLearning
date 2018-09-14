# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 07:37:10 2018

@author: benpski


Want a more unified interface to the test configs because there is a decent bit of duplication 

Also, would be useful to have some general utilities for keeping the directories organized and have the possibility of cleanup

"""
#these are fairly slow imports, would like to rearrange the order, but it crashes when tensorflow isn't imported early
from network_utils import generateAgent
import EnvironmentWrappers as EW
from manipAndWorkspace import manips, workspaces

import configparser
import os.path
import numpy as np
import uuid
import datetime
import pickle

import multiprocessing
import subprocess

import pathlib


class ConfigHandler(object):
    """
    The main things to deal with are:
        manipulator
        state
        task
        training
        network structure
        performance measures 
    """
    
    def __init__(self, config, config_location):
        #the input should be a setup configuration object
        #then the data should be extracted out of it 
        #that means the classmethods should be setting up the config their own way
        
        self.config = config
        self.config_location = config_location

        #load information from config
        #load manipulator stuff
        self.manip_name = self.config['Manipulator']['manip']
        self.manip = manips[self.manip_name]
        self.staticWorkspace = workspaces[self.manip_name]['static']
        self.dynamicWorkspace = workspaces[self.manip_name]['dynamic']
        
        #load state
        self.state = self.config['State']['state']
        self.tip = self.state == 'tip'
        
        #load training info
        self.samples = int(self.config['Training']['samples'])
        self.train_steps = int(self.config['Training']['steps'])
        self.train_bound = float(self.config['Training']['bound'])
        
        #load task, also loading task dependent params
        self.task = self.config['Task']['task']
        if self.task == 'dynReach':
            self.train_terminal = float(self.config['Training']['terminal'])
            self.test_terminal = float(self.config['Test']['terminal'])
            self.target = self.config['Task']['target']
            self.target = np.array(list(map(float,self.target.strip('[').strip(']').split())))
            self.env = EW.DynamicReaching(self.manip, self.target, tipOnly = self.tip, bound = self.train_bound, terminal = self.train_terminal)

        elif self.task == 'varTarg':
            self.train_terminal = float(self.config['Training']['terminal'])
            self.test_terminal = float(self.config['Test']['terminal'])
            self.env = EW.VariableTarget(self.manip, self.dynamicWorkspace, tipOnly = self.tip, bound = self.train_bound, terminal = self.train_terminal)
            
        elif self.task == 'varTraj':
            self.tau = float(self.config['Task']['tau'])
            self.env = EW.VariableTrajectory(self.manip, self.staticWorkspace, self.tau, tipOnly = self.tip, bound = self.train_bound)
            
        else:
            raise ArgumentError('Task name not recognized')
            
        
        #load networks
        self.actor_hidden = self.config['Networks']['actor_hidden']
        self.actor_hidden = list(map(int,self.actor_hidden.strip('[').strip(']').split(','))) #bit of a hacky way to be parsing the lists
        self.actor_act = self.config['Networks']['actor_act']
        self.critic_hidden = self.config['Networks']['critic_hidden']
        self.critic_hidden = list(map(int,self.critic_hidden.strip('[').strip(']').split(',')))
        self.critic_act = self.config['Networks']['critic_act']

        self.agent = generateAgent(self.env,self.actor_hidden,self.actor_act,self.critic_hidden,self.critic_act)
        self.weights_location = self.config['Networks']['weights_location']
        #have to check if weights exist by checking either the actor or critic weights
        if os.path.isfile(self.weights_location[:-4]+'_actor.h5f'):
            self.agent.load_weights(self.weights_location)
        
        #load test info
        self.test_steps = int(self.config['Test']['steps'])
        self.perf_location = self.config['Test']['perf_location']
        if os.path.isfile(self.perf_location):
            with open(self.perf_location,'rb') as f:
                self.perf = pickle.load(f)
        else:
            self.perf = None
                
            
    @classmethod
    def setupNew(cls,manip,state,task,training_info,testing_info,network_info):
        """
        generate a new config from the given parameters
        """
       
        config = configparser.ConfigParser()
        
        #manip
        config['Manipulator']= {'manip':manip}
        
        #state
        config['State'] = {'state':state}
        
        #task, training, and testing
        if task == 'dynReach':
            target = getTarget(workspaces[manip]['static'], workspaces[manip]['dynamic'])
            config['Task'] = {'target': target, 'task': task}
            config['Training'] = {'samples': training_info['samples'], 'steps':training_info['steps'], 'bound':training_info['bound'], 'terminal':training_info['terminal']}
            config['Test'] = {'steps': testing_info['steps'], 'terminal':testing_info['terminal']}
        elif task == 'varTarg':
            config['Task'] = {'task':task}
            config['Training'] = {'samples': training_info['samples'], 'steps':training_info['steps'], 'bound':training_info['bound'], 'terminal':training_info['terminal']}
            config['Test'] = {'steps': testing_info['steps'], 'terminal':testing_info['terminal']}
        elif task == 'varTraj':
            tau = manips[manip].dt*training_info['steps'] #need to generalize somehow
            config['Task'] = {'tau':tau, 'task':task}
            config['Training'] = {'samples': training_info['samples'], 'steps':training_info['steps'], 'bound':training_info['bound']}
            config['Test'] = {'steps': testing_info['steps']}
        else:
            raise ArgumentError("Don't recongnize the task "+task)
            
        #training
        #config['Training'] = {'samples': training_info['samples'], 'steps':training_info['steps'], 'bound':training_info['bound'], 'terminal':training_info['terminal']}
            
        #network
        config['Networks'] = {'actor_hidden': network_info['actor_hidden'],
                              'actor_act': network_info['actor_act'],
                              'critic_hidden': network_info['critic_hidden'],
                              'critic_act': network_info['critic_act']}

        
        #generate the file locations, save config with time stamp, give weights and perf the same uuid
        
        base_name = manip+'_'+state+'_'+task
        #config_location = './tests/' + base_name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.ini'
        #haven't decided a config location yet, do that when saving
        config_location = None
                    
            
        uu = str(uuid.uuid4())
        weights_location = './weights/' + base_name + '_' + uu + '.h5f'
        perf_location = './perfs/' + base_name + '_' + uu + '.pkl'
        
        config['Networks']['weights_location'] = weights_location
        
        config['Test']['perf_location'] = perf_location
        
        return cls(config,config_location)
        
    
    @classmethod
    def fromFile(cls,config_file):
        """
        load the config from a file
        """
        config = configparser.ConfigParser()
        #check file exists
        if os.path.isfile(config_file):
            config.read(config_file)
        else:
            #is this the right way of handling errors?
            raise ArgumentError("The given config file does not exist")
        
        return cls(fix_config(config),config_file)
    
    def save(self,weights=False,perf=False):
        #save the relevant data
        
        #make sure that the necessary folders exist
        try:
            os.mkdir('tests')
        except:
            pass
        try:
            os.mkdir('weights')
        except:
            pass
        try:
            os.mkdir('perfs')
        except:
            pass
        
        #save the config, may be excessive at some point
        #if a location has yet to be established for the config generate a location
        #want to make sure the timestamp is unique since there could be a collision
        #I have no idea if this will actually deal with the name collision, but it'll be close
        #better bet could be saving files in a seperate location from each other and then moving them all back together, but that likely requires a much more defined manager of configs
        if self.config_location is None:
            base_name = self.manip_name+'_'+self.state+'_'+self.task
            self.config_location = './tests/' + base_name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.ini'
            touched = False
            while not touched:
                p = pathlib.Path(self.config_location)
                try:
                    p.touch(exist_ok=False)
                    touched = True
                except:
                    self.config_location = './tests/' + base_name + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.ini'
        with open(self.config_location,'w') as f:
            self.config.write(f)
            
        if weights:
            self.agent.save_weights(self.weights_location,overwrite=False)
        
        if perf:
            with open(self.perf_location,'wb') as f:
                pickle.dump(self.perf,f)
                
        return self #incase want to iterate more
            
    def run_train(self,save=False):
        #simply run the fitting process
        self.agent.fit(self.env, nb_steps=self.samples, visualize=False, verbose=2, nb_max_episode_steps=self.train_steps)

        self.save(weights=save)

        return self 
    
    def run_test(self, save=False, render=False):
        #run the tests
        
        if self.task == 'dynReach':

            #for dynamic reaching just want to see if the target is reached
            self.env.terminal = self.test_terminal
            obs = self.env.reset()
            if render:
                self.env.render()
            reached = False
            for _ in range(self.test_steps):
                a = self.agent.forward(obs)
                obs, reward, terminal, _ = self.env.step(a)
                if render:
                    self.env.render()
                if terminal:
                    reached = True
                    break

            self.perf = reached

        elif self.task == 'varTarg':

            #for variable target want to try several points to see how it performs
            self.env.terminal = self.test_terminal
            reaches = 0
            errs = []
            for sample in range(100):
                target = self.dynamicWorkspace.sample()
                reached = False
                obs = self.env.reset()
                obs[:3] = target
                self.env.setGoal(target)
                if render:
                    self.env.render()
                for _ in range(self.test_steps):
                    a = self.agent.forward(obs)
                    obs, reward, terminal , _ = self.env.step(a)
                    if render:
                        self.env.render()
                    if terminal:
                        reaches += 1
                        reached = True
                        break
                #measure all tip errors
                pos_err = np.linalg.norm(obs[:3]-obs[6:9]) #the last tip error
                errs.append(pos_err/0.04) #normalized by length

            self.perf = (reaches,np.mean(errs))

        elif self.task == 'varTraj':

            #want to see how well a circle can be tracked

            z,r = selectTrajectoryParams(self.staticWorkspace)
            steps = self.test_steps
            w = 2*np.pi/steps
            circle = lambda t: np.array([r*np.cos(w*t),r*np.sin(w*t),z])
            circle_vel = lambda t: np.array([-r*w*np.sin(w*t),w*r*np.cos(w*t),0])
            self.env.manualGoal = True
            obs = self.env.reset()
            if render:
                self.env.render()
            errs = []
            for t in range(steps):
                pos = circle(t)
                vel = circle_vel(t)
                goal = np.concatenate([pos,vel])
                self.env.setGoal(goal)
                obs[:6] = goal

                a = self.agent.forward(obs)
                obs, _ ,_ , _ = self.env.step(a)
                if render:
                    self.env.render()
                errs.append(np.linalg.norm(goal[:3]-obs[6:9])/0.04)

            self.perf = errs

        self.save(perf=save)

        return self
            
        
    
    def _fix_locations(self):
        #in case the file locations aren't configured quite right, they can be normalized
        #really only applies for loading from a file and if weights or perf location doesn't exist
        
        weights_location = self.weights_location
        perf_location = self.perf_location
        base_name = self.manip_name+'_'+self.state+'_'+self.task
        
        if weights_location == '' and perf_location == '':
            #both need a name
            uu = str(uuid.uuid4())
            self.weights_location = './weights/' + base_name + '_' + uu + '.h5f'
            self.perf_location = './perfs/' + base_name + '_' + uu + 'pkl'
            self.config['Networks']['weights_location'] = self.weights_location
            self.config['Test']['perf_location'] = self.perf_location
        elif weights_location == '': #likely won't happen
            #get uuid from perf
            uu = perf_location.strip('.pkl').split('_')[-1]
            self.weights_location = './weights/' + base_name + '_' + uu + '.h5f'
            self.config['Networks']['weights_location'] = self.weights_location
        elif perf_location == '':
            #get uuid from weights
            uu = weights_location.strip('.h5f').split('_')[-1]
            self.perf_location = './perfs/' + base_name + '_' + uu + '.pkl'
            self.config['Test']['perf_location'] = self.perf_location
        self.save()
        return self
               
def fill_empty(config):
    #in case some values are missing from the config, fill them in with default values
    #mostly just useful for reading from a file
    #since it is really only relevant to the training section just worry about that
    if 'Training' not in config:
        config['Training'] = {'samples':0, 'steps':0}
    return config

def fix_config(config):
    #the configs keep changing and need to be updated to interface properly
    #for this, fix the training section to have the proper steps, samples, bound, and terminal
    #fix the test section to have the proper steps and terminal
    
    if config['Manipulator']['manip'] == 'cable':
        steps = 100
    else:
        steps = 200
    
    if config['Task']['task'] == 'varTraj':
        config['Training']['bound'] = str(30/100)
        config['Training']['steps'] = str(steps)
        config['Training']['samples'] = str(1000000)
        
        config['Test']['steps'] = str(500)
        
    elif config['Task']['task'] == 'varTarg':
        config['Training']['samples'] = str(200000)
        config['Training']['steps'] = str(steps)
        config['Training']['bound'] = str(30/100)
        config['Training']['terminal'] = str(2/100)
        
        config['Test']['steps'] = str(steps)
        config['Test']['terminal'] = str(5/100)
        
    else:
        config['Training']['samples'] = str(50000)
        config['Training']['steps'] = str(steps)
        config['Training']['bound'] = str(30/100)
        config['Training']['terminal'] = str(2/100)
        
        config['Test']['steps'] = str(steps)
        config['Test']['terminal'] = str(5/100)
        
    return config
        
        
    
def getTarget(staticWorkspace, dynamicsWorkspace, perturb=1e-3):
    """
    grab a point between the static and dynamic workspace
    randomly select static vertex and perturb it, see if it is between the workspaces, repeat if not
    """
    
    vertices = staticWorkspace.vertices
    
    while True:
        i = np.random.randint(0,vertices.shape[0])
        vert = vertices[i,:]
        point = vert + np.random.uniform(-1*np.array(3*[perturb]), np.array(3*[perturb]))
        
        if (not staticWorkspace.inside(point)) and dynamicsWorkspace.inside(point):
            return point

def selectTrajectoryParams(staticWorkspace):
    """
    this will be the z and the r values for the trajectory tracking
    due to the nature of the trajectories generated want to be inside the staticWorkspace
    """

    z = (staticWorkspace.max_z + staticWorkspace.min_z)/2
    r = staticWorkspace.max_r

    #want to check that the circle formed is within the boundaries of the static workspace, sample 6 rotationally symmetric points

    inside = False
    while not inside:
        check = []
        for i in range(6):
            a = 2*np.pi/6*i
            p = np.array([r*np.cos(a),r*np.sin(a),z])
            check.append(staticWorkspace.inside(p))
        inside = all(check)
        if not inside:
            r *= 0.9

    return (z, r)
        

def test_process(f):
    cmd = ['python','Test.py', f]
    subprocess.run(cmd)

def getPerformance(manip,state,task,render=False):
    """
    Of the saved policies for the relevant setup test the performances and collect them as a group
    
    search through tests for the relevant configs
    run the tests
    collect performances
    """
    name = manip+'_'+state+'_'+task
    files = ['./tests/'+i for i in os.listdir('tests/') if name in i]
    #perfs = []
    #for f in files:
    #    tester = ConfigHandler.fromFile('./tests/'+f)
    #    perf = tester.run_test(save=True,render=render).perf
    #    perfs.append(perf)
    #return perfs
    
    with multiprocessing.Pool() as p:
        p.map(test_process,files)
    