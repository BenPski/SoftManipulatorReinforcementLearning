# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 14:19:38 2018

@author: benpski


The controller for the experiments
loads the "heavier" things
    manipulators
    workspaces
    agent
    environment
    
loads from a config handler
"""

import tensorflow #an attempt to gaurantee tensorflow loads before numpy, seems to be an issue with my windows setup
import numpy as np

import Manipulators as M
import EnvironmentWrappers as EW
from manipAndWorkspace import workspaces
from network_utils import generateAgent

from ConfigHandler import TaskDynamicReaching, TaskVariableTarget, TaskVariableTrajectory


def manipFromConfig(manipConfig):
    if manipConfig.manipType.name == 'cable':
        return M.CableManipulator(manipConfig.n, manipConfig.dt, manipConfig.max_q)
    elif manipConfig.manipType.name == 'tca':
        return M.TCAManipulator(manipConfig.n, manipConfig.dt, manipConfig.max_q)
    else:
        raise Exception("Didn'rt recognize the manipulator type")

class Controller(object):
    
    def __init__(self,configHandler):
        #load all the stuff given the config handler
        self.configHandler = configHandler
        
        self.manip = manipFromConfig(self.configHandler.manip)
        self.task = self.configHandler.task #slightly useful
        self.network = self.configHandler.network
        self.staticWorkspace = workspaces[self.configHandler.manip.manipType.name]['static']
        self.dynamicWorkspace = workspaces[self.configHandler.manip.manipType.name]['dynamic']
        
        #have to check if the point for the dynamic reaching has been generated or not
        if type(self.task.taskType) == TaskDynamicReaching:
            if self.task.taskType.point is None:
                self.task.taskType.point = getTarget(self.staticWorkspace,self.dynamicWorkspace)     
        
        self.env = self.getEnv()
        self.agent = generateAgent(self.env,self.network.actor.hiddens,self.network.actor.act,self.network.critic.hiddens,self.network.critic.act)
        if self.configHandler.agent_exists():
            self.configHandler.load_agent(self.agent)
        
        self.perf = None #?
        
        
        
        
    def run_training(self,save=False):
        self.agent.fit(self.env, nb_steps=self.task.taskType.train.samples, visualize=False, verbose=2, nb_max_episode_steps=self.task.taskType.train.steps)
        if save:
            self.save(weights=True)
        return self
    
    def run_testing(self, save=False, render=False):
        perf = self.task.taskType.test.test(self.env, self.agent, self.staticWorkspace, self.dynamicWorkspace,render=render)
        self.perf = perf
        if save:
            self.save(perf=True)
        return self
    
    def save(self,weights=False,perf=False):
        
        self.configHandler.save_config()
        if weights:
            self.configHandler.save_agent(self.agent)
        if perf:
            self.configHandler.save_perf(self.perf)
            
        return self
    
    def getEnv(self):
        if type(self.task.taskType) == TaskDynamicReaching:
            return EW.DynamicReaching(self.manip, self.task.taskType.point, measure_states = self.task.measure, bound = self.task.taskType.train.bound, terminal = self.task.taskType.train.terminal)
        elif type(self.task.taskType) == TaskVariableTarget:
            return EW.VariableTarget(self.manip, self.dynamicWorkspace, measure_states = self.task.measure, bound = self.task.taskType.train.bound, terminal = self.task.taskType.train.terminal)
        elif type(self.task.taskType) == TaskVariableTrajectory:
            return EW.VariableTrajectory(self.manip, self.staticWorkspace, self.configHandler.manip.dt*self.task.taskType.train.steps, measures_states = self.task.measure, bound = self.task.taskType.train.bound)
        else:
            raise Exception("Task is of the wrong type")
    
    
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
