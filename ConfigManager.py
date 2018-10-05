# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 07:14:04 2018

@author: benpski



An interface for managing the confighandlers

Want most operations to return a subset of a config manager rather than a list of confighandlers

should it deal with controllers as well as handlers?

"""
from ConfigHandler import ConfigHandler
import ConfigHandler as CH
from os import listdir
from os.path import isfile, join


class ConfigManager(object):
    
    def __init__(self,configs):
        self.configs = configs
        
    @classmethod
    def fromDirectory(cls, directory):
        """
        Initialize the manager by grabbing all the configs in a directory
        """
        
        configs = []
        
        for f in listdir(directory):
            f = join(directory,f)
            if isfile(f):
                configs.append(ConfigHandler.fromFile(f))
        return cls(configs)
        
        pass
    
    @classmethod
    def fromFiles(cls, files):
        """
        Initialize the manager from a list of file locations
        """
        configs = []
        for f in files:
            configs.append(ConfigHandler.fromFile(f))
        return cls(configs)
        
    def __len__(self):
        return len(self.configs)
    
    def _getBy(self,pred):
        """
        Given a predicate grab the files that match
        essentially is filter
        """
        return ConfigManager([x for x in self.configs if pred(x)])
    
    def _getByManip(self,manipType):
        return self._getBy(lambda x: type(x.manip.manipType) == manipType)
        
    def _getByTask(self, taskType):
        return self._getBy(lambda x: type(x.task.taskType) == taskType)
    
    def getCables(self):
        return self._getByManip(CH.ManipCable)
    
    def getTCAs(self):
        return self._getByManip(CH.ManipTCA)
    
    def getDynReach(self):
        return self._getByTask(CH.TaskDynamicReaching)
    
    def getVarTarg(self):
        return self._getByTask(CH.TaskVariableTarget)
    
    def getVarTraj(self):
        return self._getByTask(CH.TaskVariableTrajectory)
    
    def noTraining(self):
        return self._getBy(lambda x: not x.agent_exists())
    
    def noTesting(self):
        return self._getBy(lambda x: not x.perf_exists())
        
    def _getAll(self,access):
        """
        Given a way to access something collect all of that data
        """
        return [access(x) for x in self.configs]
    
    def allPerfs(self):
        return self._getAll(lambda x: x.get_perf())
    
    def allPerfLocations(self):
        return self._getAll(lambda x: x.storage.perf_location())
    
    def allWeightLocations(self):
        return self._getAll(lambda x: x.storage.weights_location())
    
    def allConfigLocations(self):
        return self._getAll(lambda x: x.storage.config_location())
    
    
    
# some useful functions
def actual_weights(weights_location):
    #the actual weights files are split into actor and critic, but accessed with one file
    #want to actually use those names
    actor = weights_location[:-4] + '_actor.h5f'
    critic = weights_location[:-4] + '_critic.h5f'
    return (actor, critic)        

def orphanedPerfFiles():
    #look through the perf directory and collect the files
    #load the configs from tests
    #whatever is in perf and not associated with a config is orphaned
    
    CM = ConfigManager.fromDirectory('./tests/')
    perfs = [join('./perfs/',x) for x in listdir('./perfs/') if isfile(join('./perfs/',x))]
    
    CM_perfs = CM.allPerfLocations()
    
    return [x for x in perfs if x not in CM_perfs]

def orphanedWeightFiles():
    #same as orphaned perf, but for the weights files
    
    CM = ConfigManager.fromDirectory('./tests/')
    weights = [join('./weights/',x) for x in listdir('./weights/') if isfile(join('./weights/',x))]
    
    CM_weights = CM.allWeightLocations()
    #have to modify the names a bit
    act_weights = [y for x in CM_weights for y in actual_weights(x)]
        
    return [x for x in weights if x not in act_weights]

def dynReachPerfs():
    #get the performances for the dynamic reaching tasks
    
    CM = ConfigManager.fromDirectory('./tests/')
    dynReach = CM.getDynReach()
    
    perfs = dynReach.allPerfs()
    
    return perfs
    
def varTargPerfs():
    CM = ConfigManager.fromDirectory('./tests/')
    return CM.getVarTarg().allPerfs()

def varTrajPerfs():
    return ConfigManager.fromDirectory('./tests/').getVarTraj().allPerfs()

def unreadableConfigs():
    #got through the configs and find the ones that can't be read
    unread = []
    for f in listdir('./tests/'):
        f = join('./tests/',f)
        if isfile(f):
            try:
                ConfigHandler.fromFile(f)
            except:
                unread.append(f)
    return unread
    
def readableConfigs():
    read = []
    for f in listdir('./tests/'):
        f = join('./tests/',f)
        if isfile(f):
            try:
                ConfigHandler.fromFile(f)
                read.append(f)
            except:
                pass
    return read
    