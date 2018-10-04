# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 13:12:52 2018

@author: benpski

the basic config handler, only manages the config.
it would have to be promoted to a more complex controller to do testing and training

allows for the slow downs, due to some of the imports, to be avoided as the "heavy" stuff is not used

"""

import configparser
import os
import pickle
import numpy as np #could be somethign to avoid
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

def stringifyDict(d):
    #convert all the items to strings
    d_new = d.copy()
    for key, value in d.items():
        d_new[key] = str(value)
    return d_new


#these definitions came from playing around with haskell types, probably not the greatest way of doing it
#the product types are mapped to classes of multiple arguments in their constructor (they essentially are just acting as dicts)
#the sum types map to several classes with a unified superclass

#Manipulator
class ManipType(object): pass
class ManipCable(ManipType): 
    def __init__(self):
        self.name = 'cable'
class ManipTCA(ManipType): 
    def __init__(self):
        self.name = 'tca'

class ManipConfig():
    def __init__(self,n,dt,max_q,manipType):
        self.n = n
        self.dt = dt
        self.max_q = max_q
        self.manipType = manipType
    @classmethod
    def fromDict(cls,info):
        n = int(info['n'])
        dt = float(info['dt'])
        max_q = float(info['max_q'])
        manipType = info['manipType']
        
        if manipType == 'cable':
            manipType = ManipCable()
        elif manipType == 'tca':
            manipType = ManipTCA()
        else:
            raise Exception("The type of manipulator is unrecognized")
            
        return cls(n,dt,max_q,manipType)
    
    def getConfig(self):
        return {'n': self.n, 'dt': self.dt, 'max_q': self.max_q, 'manipType': self.manipType.name}

#networks
class Network():
    def __init__(self, act_fun, hiddens):
        self.act = act_fun
        self.hiddens = hiddens
    def getConfig(self):
        return self.__dict__
class RLNetworks():
    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic
    @classmethod
    def fromDict(cls, info):
        actor_act = info['actor_act']
        critic_act = info['critic_act']
        readHidden = lambda h: list(map(int, h.strip('[').strip(']').split(',')))
        actor_hiddens = readHidden(info['actor_hiddens'])
        critic_hiddens = readHidden(info['critic_hiddens'])
        
        actor = Network(actor_act, actor_hiddens)
        critic = Network(critic_act, critic_hiddens)
        
        return cls(actor, critic)
    
    def getConfig(self):
        return combineDict(appendToKeys('actor_', self.actor.getConfig()), appendToKeys('critic_', self.critic.getConfig()))
        
#storage
class Storage():
    def __init__(self,uuid):
        self.uuid = uuid
        
    @classmethod
    def fromDict(cls,info):
        return cls(info['uuid'])
        
    def config_location(self):
        return "./tests/" + self.uuid + ".ini"
    
    def perf_location(self):
        return "./perfs/" + self.uuid + ".pkl"
    
    def weights_location(self):
        return "./weights/" + self.uuid + ".h5f"
    
    def getConfig(self):
        return self.__dict__
    

#task
class TaskTest(): 
    def getConfig(self):
        return appendToKeys('test_', self.__dict__)
class TaskTrain():
    def getConfig(self):
        return appendToKeys('train_', self.__dict__)
    
class DynReachTrain(TaskTrain):
    def __init__(self, steps, samples, bound, terminal):
        self.steps = steps
        self.samples = samples
        self.bound = bound
        self.terminal = terminal
class DynReachTest(TaskTest):
    def __init__(self, steps, terminal):
        self.steps = steps
        self.terminal = terminal
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
class VarTargTest(TaskTest):
    def __init__(self, steps, repeat, terminal):
        self.steps = steps
        self.repeat = repeat
        self.terminal = terminal
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
class VarTrajTest(TaskTest):
    def __init__(self, steps):
        self.steps = steps
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
        
    @classmethod
    def fromDict(cls,info):
        measure = int(info['measure'])
        taskType = info['task']
        
        if taskType == 'dynReach':
            train_steps = int(info['train_steps'])
            train_samples = int(info['train_samples'])
            train_bound = float(info['train_bound'])
            train_terminal = float(info['train_terminal'])
            test_steps = int(info['test_steps'])
            test_terminal = float(info['test_terminal'])
            #it is possible that point is none
            if info['point'] == 'None':
                point = None
            else:
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
        return cls(measure, task)
    
    def getConfig(self):
        #print(combineDict({'measure': self.measure}, self.taskType.getConfig()))
        return stringifyDict(combineDict({'measure': str(self.measure)}, self.taskType.getConfig()))


class ConfigHandler(object):
    """
    Handles the configs for the rl-experiments
    adds a bit of functionality without the heavy processes
    
    loads from either an existing config or defined by some parameters
    
    the goal is to only define all the necessary information, but not to load it
    """
    
    def __init__(self,config):
        self.config = config
        
        self.manip = ManipConfig.fromDict(config['Manipulator'])
        self.task = Task.fromDict(config['Task'])
        self.network = RLNetworks.fromDict(config['Network'])
        self.storage = Storage.fromDict(config['Locations'])
        
    @classmethod
    def fromFile(cls,config_location):
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
        
        return cls(config)
    
    @classmethod
    def setup(cls,act_type, task_type,state_measure,train_steps,train_samples,train_bound,test_steps, actor_act,actor_hidden,critic_act,critic_hidden, **kwargs):
        """
        Setting up a config from given parameters
        
        assumes some default values
        
        kwargs are necessary for the variability in the task specification
        """
        if act_type == 'cable':
            manip = ManipConfig(10,0.01,0.1,ManipCable())
        elif act_type == 'tca':
            manip = ManipConfig(5,0.5,3,ManipTCA())
        else:
            raise Exception("Didn't recognize the given actuator type")
            
        if task_type == 'dynReach':
            train_terminal = kwargs['train_terminal']
            test_terminal = kwargs['test_terminal']
            train = DynReachTrain(train_steps, train_samples, train_bound, train_terminal)
            test = DynReachTest(test_steps, test_terminal)
            #point = getTarget(manip.manipType.staticWorkspace,manip.manipType.dynamicWorkspace)
            point = None
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
        
        config = configparser.ConfigParser()
        config['Manipulator'] = manip.getConfig()
        config['Task'] = task.getConfig()
        config['Network'] = network.getConfig()
        config['Locations'] = locations.getConfig()
        
        return cls(config)
    
    def save_perf(self,perf):
        #save the performance measures to the appropriate place
        with open(self.storage.perf_location(), 'wb') as f:
            pickle.dump(perf,f)
        return self
            
    def save_agent(self,agent):
        #save the weights of the agent to the appropriate place
        agent.save_weights(self.storage.weights_location(), overwrite=True)
        return self
    
    def load_perf(self):
        #get the performance measure
        with open(self.storage.perf_location(), 'rb') as f:
            perf = pickle.load(f)
        return perf
    
    def load_agent(self,agent):
        #given th eagent load the weights
        agent.load_weights(self.storage.weights_location())
        return self
    
    def perf_exists(self):
        #check if perf file has been generated
        if os.path.isfile(self.storage.perf_location()):
            return True
        else:
            return False
        
    def agent_exists(self):
        #check if there are saved weights for the agent
        #have to amend file location slightly
        weights_location = self.storage.weights_location()
        actor = weights_location[:-4] + '_actor.h5f'
        critic = weights_location[:-4] + '_critic.h5f'
        if os.path.isfile(actor) and os.path.isfile(critic):
            return True
        else:
            return False
        
    def save_config(self):
        #save the config, can no longer provide generic saving since there is no access to the needed information
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
        config_location = self.storage.config_location()
        with open(config_location,'w') as f:
            self.config.write(f)
                
        return self #incase want to iterate more
   
#some useful functions for workspace stuff
        
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

    