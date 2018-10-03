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

#fixes something
import tensorflow #tensorflow has to get imported before numpy for some reason
import numpy as np
from network_utils import generateAgent

from abc import ABCMeta,abstractmethod
from manipAndWorkspace import workspaces
import Manipulators as M
import EnvironmentWrappers as EW
import configparser
import os
import pickle
import uuid





#will be useful for combining configs        
def combineDict(a,b):
    z = a.copy()
    z.update(b)
    return z

def appendToKeys(name, d):
    d_new = {}
    for key, values in d.items():
        d_new[name + key] = values
    return d_new



#these definitions came from playing around with haskell types, probably not the greatest way of doing it
#the product types are mapped to classes of multiple arguments in their constructor (they essentially are just acting as dicts)
#the sum types map to several classes with a unified superclass

#Manipulator
class ManipType(object, metaclass=ABCMeta):
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
class TaskTest(object, metaclass=ABCMeta): pass
#    @abstractmethod
#    def train(self,env,agent,staticWorkspace,dynamicWorkspace,render=False):
#        pass
class TaskTrain(): pass
    
class DynReachTrain(TaskTrain):
    def __init__(self, steps, samples, bound, terminal):
        self.steps = steps
        self.samples = samples
        self.bound = bound
        self.terminal = terminal
    def getConfig(self):
        return appendToKeys('train_', self.__dict__)
class DynReachTest(TaskTest):
    def __init__(self, steps, terminal):
        self.steps = steps
        self.terminal = terminal
    def getConfig(self):
        return appendToKeys('test_', self.__dict__)
    def test(self, env, agent, staticWorkspace, dynamicWorkspace, render=False):
        env.terminal = self.terminal
        obs = env.reset()
        if render:
            env.render()
        reached = False
        for _ in range(self.steps):
            a = agent.forward(obs)
            obs, reward, terminal, _ = env.step(a)
            if render:
                env.render()
            if terminal:
                reached = True
                break
        return reached


class VarTargTrain(TaskTrain):
    def __init__(self, steps, samples, bound, terminal):
        self.steps = steps
        self.samples = samples
        self.bound = bound
        self.terminal = terminal
    def getConfig(self):
        return appendToKeys('train_', self.__dict__)
class VarTargTest(TaskTest):
    def __init__(self, steps, repeat, terminal):
        self.steps = steps
        self.repeat = repeat
        self.terminal = terminal
    def getConfig(self):
        return appendToKeys('test_', self.__dict__)
    def test(self, env, agent, _, workspace, render=False):
        env.terminal = self.terminal
        reaches = 0
        errs = []
        for sample in range(self.repeat):
            target = workspace.sample()
            obs = env.reset()
            obs[:3] = target
            env.setGoal(target)
            if render:
                env.render()
            for _ in range(self.steps):
                a = agent.forward(obs)
                obs, reward, terminal, _ = env.step(a)
                if render:
                    env.render()
                if terminal:
                    reaches += 1
                    break
            pos_err = np.linalg.norm(obs[:3]-obs[6:9])
            errs.append(pos_err/0.04)
        return (reaches, np.mean(errs))
        
    
class VarTrajTrain(TaskTrain):
    def __init__(self, steps, samples, bound):
        self.steps = steps
        self.samples = samples
        self.bound = bound
    def getConfig(self):
        return appendToKeys('train_', self.__dict__)
class VarTrajTest(TaskTest):
    def __init__(self, steps):
        self.steps = steps
    def getConfig(self):
        return appendToKeys('test_',self.__dict__)
    def test(self, env, agent, workspace, _, render = False):
        z,r = selectTrajectoryParams(workspace)
        steps = self.steps 
        w = 2*np.pi/steps
        circle = lambda t: np.array([r*np.cos(w*t),r*np.sin(w*t),z])
        circle_vel = lambda t: np.array([-r*w*np.sin(w*t),w*r*np.cos(w*t),0])
        env.manualGoal = True
        obs = env.reset()
        if render:
            env.render()
        errs = []
        for t in range(steps):
            pos = circle(t)
            vel = circle_vel(t)
            goal = np.concatenate([pos,vel])
            env.setGoal(goal)
            obs[:6] = goal
            a = agent.forward(obs)
            obs, _, _, _ = env.step(a)
            if render:
                env.render()
            errs.append(np.linalg.norm(goal[:3]-obs[6:9])/0.04)
        return errs


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
        
def manipFromConfig(info):
    n = int(info['n'])
    dt = float(info['dt'])
    max_q = float(info['max_q'])
    manipType = info['manipType']
    if manipType == 'cable':
        return Manip(n,dt,max_q,ManipCable())
    elif manipType == 'tca':
        return Manip(n,dt,max_q,ManipTCA())
    else:
        raise Exception("Unrecognized manipType in config")
        
def taskFromConfig(info):
    measure = int(info['measure'])
    taskType = info['task']
    
    if taskType == 'dynReach':
        train_steps = int(info['train_steps'])
        train_samples = int(info['train_samples'])
        train_bound = float(info['train_bound'])
        train_terminal = float(info['train_terminal'])
        test_steps = int(info['test_steps'])
        test_terminal = float(info['test_terminal'])
        point = np.array(list(map(float, info['point'].strip('[').strip(']').split())))
        train = DynReachTrain(train_steps, train_samples, train_bound, train_terminal)
        test = DynReachTest(test_steps, test_terminal)
        task = TaskDynamicReaching(point, train, test)
    elif taskType == 'varTarg':
        train_steps = int(info['train_steps'])
        train_samples = int(info['train_samples'])
        train_bound = float(info['train_bound'])
        train_terminal = float(info['train_terminal'])
        test_steps = int(info['test_steps'])
        test_repeat = int(info['test_repeat'])
        test_terminal = float(info['test_terminal'])
        train = VarTargTrain(train_steps, train_samples, train_bound, train_terminal)
        test = VarTargTest(test_steps, test_repeat, test_terminal)
        task = TaskVariableTarget(train, test)
    elif taskType == 'varTraj':
        train_steps = int(info['train_steps'])
        train_samples = int(info['train_samples'])
        train_bound = float(info['train_bound'])
        test_steps = int(info['test_steps'])
        train = VarTrajTrain(train_steps, train_samples, train_bound)
        test = VarTrajTest(test_steps)
        task = TaskVariableTrajectory(train,test)
    else:
        raise Exception("Unrecognized task in config")
        
    return Task(measure, task)
    
def stringToActFun(name):
    if name == 'relu':
        return Relu()
    elif name == 'tanh':
        return Tanh()
    elif name == 'sigmoid':
        return Sigmoid()
    else:
        raise Exception("Unrecognized act_fun")

def networkFromConfig(info):
    actor_act = info['actor_act']
    critic_act = info['critic_act']
    readHiddens = lambda h: list(map(int,h.strip('[').strip(']').split(',')))
    actor_hiddens = readHiddens(info['actor_hiddens'])
    critic_hiddens = readHiddens(info['critic_hiddens'])
    
    
    actor = Network(stringToActFun(actor_act), actor_hiddens)
    critic = Network(stringToActFun(critic_act), critic_hiddens)
    
    return RLNetworks(actor,critic)

def locationsFromConfig(info):
    return Storage(info['uuid'])

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
        
        #self.agent = None #want to avoid loading it until necessary
        self.agent = self.getAgent() #not possible to avoid some of the slow down
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
        
        manip = manipFromConfig(config['Manipulator'])
        task = taskFromConfig(config['Task'])
        network = networkFromConfig(config['Network'])
        locations = locationsFromConfig(config['Locations'])
        
        return cls(manip, task, network, locations)
    
    @classmethod
    def setup(cls,act_type, task_type,state_measure,train_steps,train_samples,train_bound,test_steps, actor_act,actor_hidden,critic_act,critic_hidden, **kwargs):
        """
        Setting up a config from given parameters
        
        assumes some default values
        
        kwargs are necessary for the variability in the task specification
        """
        if act_type == 'cable':
            manip = Manip(10,0.01,0.1,ManipCable())
        elif act_type == 'tca':
            manip = Manip(5,0.5,3,ManipTCA())
        else:
            raise Exception("Didn't recognize the given actuator type")
            
        if task_type == 'dynReach':
            train_terminal = kwargs['train_terminal']
            test_terminal = kwargs['test_terminal']
            train = DynReachTrain(train_steps, train_samples, train_bound, train_terminal)
            test = DynReachTest(test_steps, test_terminal)
            point = getTarget(manip.manipType.staticWorkspace,manip.manipType.dynamicWorkspace)
            task = TaskDynamicReaching(point,train,test)
        elif task_type == 'varTarg':
            train_terminal = kwargs['train_terminal']
            test_repeat = kwargs['test_repeat']
            test_terminal = kwargs['test_terminal']
            train = VarTargTrain(train_steps, train_samples, train_bound, train_terminal)
            test = VarTargTest(test_steps, test_repeat, test_terminal)
            task = TaskVariableTarget(train,test)
        elif task_type == 'varTraj':
            train = VarTrajTrain(train_steps, train_samples, train_bound)
            test = VarTrajTest(test_steps)
            task = TaskVariableTrajectory(train,test)
        else:
            raise Exception("Didn't recognize the given task type")
            
        task = Task(state_measure, task)
            
        actor = Network(actor_act, actor_hidden)
        critic = Network(critic_act, critic_hidden)
        network = RLNetworks(actor, critic)
        
        #generate uuid
        uu = str(uuid.uuid4())
        locations = Storage(uu)
        
        return cls(manip,task,network,locations)
    
    def getEnv(self):
        if type(self.task.taskType) == TaskDynamicReaching:
            return EW.DynamicReaching(self.manip.manip, self.task.taskType.point, measure_states = self.task.measure, bound = self.task.taskType.train.bound, terminal = self.task.taskType.train.terminal)
        elif type(self.task.taskType) == TaskVariableTarget:
            return EW.VariableTarget(self.manip.manip, self.manip.manipType.dynamicWorkspace, measure_states = self.task.measure, bound = self.task.taskType.train.bound, terminal = self.task.taskType.train.terminal)
        elif type(self.task.taskType) == TaskVariableTrajectory:
            return EW.VariableTrajectory(self.manip.manip, self.manip.manipType.staticWorkspace, self.manip.dt*self.task.taskType.train.steps, measures_states = self.task.measure, bound = self.task.taskType.train.bound)
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
                pickle.dump(self.perf,f)
                
        return self #incase want to iterate more

    def run_training(self, save = False):
        
        if self.agent is None:
            self.agent = self.getAgent()
            
        self.agent.fit(self.env, nb_steps=self.task.taskType.train.samples, visualize=False, verbose=2, nb_max_episode_steps=self.task.taskType.train.steps)

        self.save(weights=save)

        return self 
    
    def run_testing(self, save=False, render=False):
        if self.agent is None:
            self.agent = self.getAgent()
        perf = self.task.taskType.test.test(self.env, self.agent, self.manip.manipType.staticWorkspace, self.manip.manipType.dynamicWorkspace,render=render)
        self.perf = perf
        self.save(perf=save)
        return self
        
        
    def getAgent(self):
        agent = generateAgent(self.env,self.network.actor.hiddens,self.network.actor.act_fun.name,self.network.critic.hiddens,self.network.critic.act_fun.name)
        return agent
       
        
    
#some useful functions for workspace stuff
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
