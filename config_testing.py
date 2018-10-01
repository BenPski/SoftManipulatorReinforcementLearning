# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 08:48:38 2018

@author: benpski


Here looking to redesign the configuration handling

The main drive is to have a more configurable state measurement

Want to organize the structure a bit better and get some less manual read/write

would also be very helpful to get some of the load times down

"""




"""
So a configuration is defined by:
    the manipulator used
    the task 
    the training info
    the testign info
    
the manipulator is defined by
    the number of spatial points
    the time step
    the type of actuator
    
the task is defined by 
    the type of task
    any extra info necessary
    
the test info
    number of steps
    
the training
    steps
    samples 
"""

from abc import ABC,abstractmethod
from manipAndWorkspace import workspaces
import Manipulators as M
import EnvironmentWrappers as EW
import configparser
import os
import pickle


#will be useful for combining configs        
def combineDict(a,b):
    z = a.copy()
    z.update(b)
    return z


#these definitions came from playing around with haskell types, probably not the greatest way of doing it
#the product types are mapped to classes of multiple arguments in their constructor (they essentially are just acting as dicts)
#the sum types map to several classes with a unified superclass

#Manipulator
class ManipType(ABC):
    @abstractmethod
    def manip_start(self,n,dt,max_q):
        pass
class ManipCable(ManipType):
    def __init__(self):
        self.staticWorkspace = workspaces['cable']['static']
        self.dynamicWorkspace = workspaces['cable']['dynamic']
        self.name = 'cable'
    def manip_start(self,n,dt,max_q):
        return M.CableManipulator(n,dt,max_q)
class ManipTCA(ManipType): 
    def __init__(self):
        self.staticWorkspace = workspaces['tca']['static']
        self.dynamicWorkspace = workspaces['tca']['dynamic']
        self.name = 'tca'
    def manip_start(self,n,dt,max_q):
        return M.TCAManipulator(n,dt,max_q)

def manipFromString(name):
    if name == 'cable':
        return ManipCable()
    elif name == 'tca':
        return ManipTCA()
    else:
        raise Exception("The given name is not recognized for the manipulators")


#manip = Manip n dt max_q manip
class Manip():
    def __init__(self, n, dt, max_q, manipType):
        self.n = n
        self.dt = dt
        self.max_q = max_q
        self.manipType = manipType
        
        self.manip = manipType.manip_start(self.n,self.dt,self.max_q)
        
    def getConfig(self):
        #do not want to save the manipulator, just how to restart it
        return {'n':self.n,'dt':self.dt,'max_q':self.max_q,'manipType':self.manipType.name}
        

#networks
class ActFun(): pass
class Relu(ActFun): 
    def __init__(self):
        self.name = 'relu'
class Tanh(ActFun): 
    def __init__(self):
        self.name = 'tanh'
class Sigmoid(ActFun): 
    def __init__(self):
        self.name = 'sigmoid'

class Network():
    def __init__(self, act_fun, hiddens):
        self.act_fun = act_fun
        self.hiddens = hiddens
class RLNetworks():
    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic
    def getConfig(self):
        #can't rely on the Network class due to potential collisions
        return {'actor_act': self.actor.act_fun.name, 'actor_hiddens': self.actor.hiddens, 'critic_act': self.critic.act_fun.name, 'critic_hiddens': self.critic.hiddens}
        
#storage
class Storage():
    def __init__(self,uuid):
        self.uuid = uuid
        
    def config_location(self):
        return "./tests/" + self.uuid + ".ini"
    
    def perf_location(self):
        return "./perfs/" + self.uuid + ".pkl"
    
    def weights_location(self):
        return "./weights/" + self.uuid + ".h5f"
    
    def getConfig(self):
        return self.__dict__
    

#task
class TaskTest(): pass
class TaskTrain(): pass
    
class DynReachTrain(TaskTrain):
    def __init__(self, steps, samples, bound, terminal):
        self.steps = steps
        self.samples = samples
        self.bound = bound
        self.terminal = terminal
    def getConfig(self):
        return self.__dict__
class DynReachTest(TaskTest):
    def __init__(self, terminal):
        self.terminal = terminal
    def getConfig(self):
        return self.__dict__

class VarTargTrain(TaskTrain):
    def __init__(self, steps, samples, bound, terminal):
        self.steps = steps
        self.samples = samples
        self.bound = bound
        self.terminal = terminal
    def getConfig(self):
        return self.__dict__
class VarTargTest(TaskTest):
    def __init__(self, samples, repeat, terminal):
        self.samples = samples
        self.repeat = repeat
        self.terminal = terminal
    def getConfig(self):
        return self.__dict__
    
class VarTrajTrain(TaskTrain):
    def __init__(self, steps, samples, bound):
        self.steps = steps
        self.samples = samples
        self.bound = bound
    def getConfig(self):
        return self.__dict__
class VarTrajTest(TaskTest):
    def __init__(self, steps):
        self.steps = steps
    def getConfig(self):
        return self.__dict__

class TaskType(): pass
class TaskDynamicReaching(TaskType): 
    def __init__(self, point, train, test):
        self.point = point
        self.test = test
        self.train = train
    def getConfig(self):
        return combineDict({'task':'dynReach', 'point': self.point}, combineDict(self.test.getConfig(), self.train.getConfig()))
        
class TaskVariableTarget(TaskType): 
    def __init__(self, train, test):
        self.test = test
        self.train = train        
    def getConfig(self):
        return combineDict({'task':'varTarg'}, combineDict(self.test.getConfig(), self.train.getConfig()))
        
class TaskVariableTrajectory(TaskType): 
    def __init__(self, train, test):
        self.test = test
        self.train = train        
    def getConfig(self):
        return combineDict({'task':'varTraj'}, combineDict(self.test.getConfig(), self.train.getConfig()))

class Task():
    def __init__(self, measure, taskType):
        self.measure = measure
        self.taskType = taskType
        
    def getConfig(self):
        return combineDict({'measure': self.measure}, self.taskType.getConfig())
        
#total config
# it is necessary for the config to be loadable from a file or initialized given some values
# for now initializing from values is just the constructor
        


class Config():
    def __init__(self, manip, task, network, locations):
        self.manip = manip
        self.task = task
        self.network = network
        self.locations = locations
        
        self.config = configparser.ConfigParser()
        self.config['Manipulator'] = self.manip.getConfig()
        self.config['Task'] = self.task.getConfig()
        self.config['Network'] = self.network.getConfig()
        self.config['Locations'] = self.locations.getConfig()
        
        self.perf = self.getPerf()
        
        self.agent = None #want to avoid loading it until necessary
        self.env = self.getEnv()
        
    @classmethod    
    def fromFile(cls, config_location):
        """
        load from a file
        """
        config = configparser.ConfigParser()
        #check file exists
        if os.path.isfile(config_location):
            config.read(config_location)
        else:
            #is this the right way of handling errors?
            raise Exception("The given config file does not exist")
        
        manip = config['Manipulator'] 
        task = config['Task']
        network = config['Network']
        locations = config['Locations']
        
        return cls(manip, task, network, locations)
    
    def getEnv(self):
        if type(self.task) == TaskDynamicReaching:
            return EW.DynamicReaching(self.manip.manip, self.task.point, measure_states = self.task.measure, bound = self.train.bound, terminal = self.train.terminal)
        elif type(self.task) == TaskVariableTarget:
            return EW.VariableTarget(self.manip.manip, self.manip.dynamicWorkspace, measure_states = self.task.measure, bound = self.train.bound, terminal = self.train.terminal)
        elif type(self.task) == TaskVariableTrajectory:
            return EW.VariableTrajectory(self.manip.manip, self.manip.staticWorkspace, self.manip.dt*self.task.train.steps, measures_states = self.task.measure, bound = self.train.bound)
        else:
            raise Exception("Task is of the wrong type")
        
    def getPerf(self):
        #if perf file can be read return the value otherwise None
        try:
            with open(self.locations.perf_location(),'wb') as f:
                perf = pickle.load(f)
        except:
            perf = None
            
        return perf
    
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
        config_location = self.locations.config_location()
        with open(config_location,'w') as f:
            self.config.write(f)
           
        if weights:
            self.agent.save_weights(self.locations.weights_location(),overwrite=False)
        
        if perf:
            with open(self.locations.perf_location(),'wb') as f:
                pickle.dump(self.getPerf(),f)
                
        return self #incase want to iterate more

    def run_training(self, save = False):
        
        if self.agent is None:
            self.agent = self.getAgent()
            
        self.agent.fit(self.env, nb_steps=self.task.train.samples, visualize=False, verbose=2, nb_max_episode_steps=self.train.steps)

        self.save(weights=save)

        return self 
    
    def run_testing(self):
        pass
        
    def getAgent(self):
        from network_utils import generateAgent
        agent = generateAgent(self.env,self.network.actor.hidden,self.network.actor.act_fun.name,self.network.critic.hiddens,self.network.critic.act_fun)
        return agent
        
        

