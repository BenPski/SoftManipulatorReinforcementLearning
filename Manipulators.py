"""
The definitions for the interfaces to the dynamics of the manipulators


Need to specify more of the geometry from here for a more general solution
"""

from abc import ABCMeta, abstractmethod
import matlab.engine as engine
import numpy as np

from matlab_utils import mat2np, np2mat

import configparser
import os
import sys

def readConfig():
    """
    read the configuration file
    """

    #trying to load the configuration
    config = configparser.ConfigParser()
    if os.path.exists('manip_config.ini'):
        config.read('manip_config.ini')
    else:
        config['Manipulators'] = {'manip_location': "."}
        with open('manip_config.ini','w') as f:
            config.write(f)
        sys.exit("In order to connect to Matlab, need to know where the directory is where the matlab code exists forthe soft manipulators. A basic one has been created and it likely needs to be edited as it will only search the current directory.")

    #pull out the location
    try:
        location = config['Manipulators']['manip_location']
    except:
        sys.exit("The config file appears to be formatted incorrectly")

    #check if the directory exists and if the right files are found

    if os.path.isdir(location):
        #check for the files
        if os.path.exists(os.path.join(location,"initialDynamics.m")) and os.path.exists(os.path.join(location,"fastDynamicsStable.m")) and os.path.exists(os.path.join(location,"initTCADynamics.m")) and os.path.exists(os.path.join(location,"fullTCADynamics.m")):
            return location
        else:
            sys.exit("One of the required manipulator files was not found in the directory given.")
    else:
        sys.exit("The directory specified either does not exist or it is not a directory.")




class Manipulator(object, metaclass=ABCMeta):
    """
    the base interface to the manipulator dynamics

    need to know time discretization, the number of spacial points, and maximum allowed actuation
    keeps track of the state, which consists of g, xi, and eta and potentially others
        split into kinematic and other states? or just have one state
    also, since the current implementations require a matlab connection, that needs to be established first
    """

    def __init__(self,n,dt,max_q):
        self.n = n
        self.dt = dt
        self.max_q = max_q
        #self.base_location = base_location # base location for the matlab connection
        self.base_location = readConfig()
        self.eng = self.connectMatlab()

        self.state = {} #store states as a dictionary

        self.state = self.initState()


    def connectMatlab(self):
        """
        establish a connection with matlab
        this is likely a goo default definition
        """
        eng = engine.start_matlab()
        eng.cd(self.base_location)
        return eng

    @abstractmethod
    def initState(self):
        """
        should initialize the state consistently
        return the state dictionary
        """
        pass

    @abstractmethod
    def step(self,a):
        """
        step the dynamics one step forward given an action
        return the state dictionary and action
        """
        pass

    def getState(self):
        """
        grab the state of the manipulator
        probably a good default
        better as a property?
        """
        return self.state

    def render(self, hold=False):
        """
        plotting the state
        """

        if hold:
            self.eng.hold("on",nargout=0)
        g = self.getState()['g']

        self.eng.plot3(np2mat(g[:,9]),np2mat(g[:,10]),np2mat(g[:,11]))
        self.eng.daspect(np2mat([1,1,1]),nargout=0)

    def __getstate__(self):
        #can't pickle the matlab engine, so need to avoid it
        d = {}
        for key, item in self.__dict__.items():
            if key != 'eng': #remove engine
                d[key] = item
        return d

    def __setstate__(self,s):
        self.__dict__.update(s)
        #have to restart the connection
        self.eng = self.connectMatlab()



class CableManipulator(Manipulator):
    """
    implementation for  the cable driven manipulator
    """

    def initState(self):
        g, xi, eta = self.eng.initDynamics(self.n,nargout=3)
        #xi, eta, g = self.eng.initialDynamics(self.n,nargout=3)
        g = mat2np(g)
        xi = mat2np(xi)
        eta = mat2np(eta)
        state = {'g':g, 'xi': xi, 'eta':eta}

        self.state = state

        return state

    def step(self,a):
        # a goes from 0 to 1, so adjust it to be from 0 to the max
        #a = np.clip(a,0,self.max_q) #could arrange better ?
        a = np.clip(a,0,1) #have to eliminate bias
        a = self.max_q*a
        #print(a)
        eta = self.getState()['eta']
        xi = self.getState()['xi']
        g,xi,eta = self.eng.dynamicsCable(np2mat(a),np2mat(eta),np2mat(xi),self.dt,nargout=3)
        #g,xi,eta = self.eng.fastDynamicsStable(np2mat(a),np2mat(eta),np2mat(xi),self.dt,nargout=3)
        g = mat2np(g)
        xi = mat2np(xi)
        eta = mat2np(eta)

        state = {'g':g, 'xi':xi, 'eta':eta}

        self.state = state

        return (state, a)


class TCAManipulator(Manipulator):
    """
    implementation for  the tca driven manipulator
    """

    def initState(self):
        g,xi,eta,tcaTemps = self.eng.initDynamics(self.n,nargout=4)
        #g, xi, eta, tcaTemps = self.eng.initTCADynamics(self.n,nargout=4)
        g = mat2np(g)
        xi = mat2np(xi)
        eta = mat2np(eta)
        state = {'g':g, 'xi': xi, 'eta':eta, 'tcaTemps': tcaTemps}

        self.state = state

        return state

    def step(self,a):
        a = np.clip(a,0,1)
        a = self.max_q*a
        #print(a)
        eta = self.getState()['eta']
        xi = self.getState()['xi']
        tcaTemps = self.getState()['tcaTemps']
        g,xi,eta,tcaTemps = self.eng.dynamicsTCA(np2mat(a),np2mat(eta),np2mat(xi),self.dt,tcaTemps,nargout=4)
        #g,xi,eta,tcaTemps = self.eng.fullTCADynamics(np2mat(a),np2mat(eta),np2mat(xi),self.dt,tcaTemps,nargout=4)
        g = mat2np(g)
        xi = mat2np(xi)
        eta = mat2np(eta)

        state = {'g':g, 'xi':xi, 'eta':eta, 'tcaTemps':tcaTemps}

        self.state = state

        return (state, a)
