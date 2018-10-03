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

import tensorflow

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
        
        self.env = self.getEnv()
        self.agent = generateAgent(self.env,self.network.actor.hiddens,self.network.actor.act,self.network.critic.hiddens,self.network.critic.act)
        if self.configHandler.agent_exists():
            self.configHandler.load_agent(self.agent)
        
        self.perf = None #?
        
        
    def run_training(self):
        self.agent.fit(self.env, nb_steps=self.task.taskType.train.samples, visualize=False, verbose=2, nb_max_episode_steps=self.task.taskType.train.steps)
        self.save(weights=True)
        return self
    
    def run_testing(self, render=False):
        perf = self.task.taskType.test.test(self.env, self.agent, self.staticWorkspace, self.dynamicWorkspace,render=render)
        self.perf = perf
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
    
    
