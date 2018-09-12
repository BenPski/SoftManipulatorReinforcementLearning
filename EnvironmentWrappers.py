from abc import ABCMeta, abstractmethod
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from matlab_utils import np2mat, mat2np

class EnvironmentWrapper(gym.Env, metaclass=ABCMeta):
    """
    want to convert a manipulator model to a gym environment

    to be an enviornment need to implement:
        action space
        observation space
        step
        reset
        seed
        render

    then since the particular tests being done will have variable goals, termination conditions, and reward functions those need to be implemented as well
    """

    def __init__(self,manipulator,observation_space): #the manipulator should already be initialized
        self.manipulator = manipulator
        self.action_space = self.getActionSpace()
        #self.observation_space = self.getObservationSpace(), precednce issue
        self.observation_space = observation_space #have to pass in the relevant observation space


    def step(self,a):
        """
        take the dynamics step and give back the observation, reward, termination, and extra info
        """
        self.manipulator.step(a)
        state = self.manipulator.getState()
        obs = self.getObservation(state)

        reward = self.getReward(state)

        terminal = self.getTerminal(state)

        self.extraSteps()

        return obs, reward, terminal, {}

    def reset(self):
        """
        reset the state of the manipulator and environment and return an intial observation
        in order to reset want to reset both the manipulator and the environment

        in the case of the environment this could be preparing a new goal or trajectory
        for the manipulator this could be just resetting to the reference state, perturbing it slightly, or leaving the state as is
        """
        self.envReset()
        state = self.manipulatorReset()

        return self.getObservation(state)

    def seed(self, seed=None):
        """
        I still don't get the purpose of this, or if I'm doing it right
        probably just for being able to initialize with a desired seed, but I've never bothered
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        """
        do the rendering process, since it is common to have a variable goal want to draw that along with the manipulator
        """
        goal = self.getGoal()
        self.manipulator.render(hold=True)
        self.manipulator.eng.scatter3(np2mat([goal[0]]), np2mat([goal[1]]), np2mat([goal[2]]))

        if close:
            self.manipulator.eng.close("all",nargout=0)

    @abstractmethod
    def getGoal(self):
        """
        get the current goal of the environment
        a necessary method for rendering

        it should be required by getObservation and getReward functions
        """
        pass

    @abstractmethod
    def envReset(self):
        """
        reset the environment
        this should typically be resetting the goal and possibly other counts of things
        """
        pass

    @abstractmethod
    def manipulatorReset(self):
        """
        reset the manipulator and return the initial state

        this can be a variety of things based off of how it is desired to progress with training
            return to reference state
            return to reference and perturb slightly
            stay at current state (goes with a variable goal)
        """
        pass


    def getActionSpace(self,acts = 3):
        """
        get the action space for the manipulator
        usually this will be [0,0,0] to [max_q,max_q,max_q] because they should have 3 actuators going from 0 to max_q
        """
        max_q = self.manipulator.max_q

        action_space = spaces.Box(low = np.array([0,0,0]), high = max_q*np.array([1,1,1]), dtype=np.float32)
        return action_space


    @abstractmethod
    def getObservation(self,state):
        """
        given a manipulator state return the observation
        """
        pass

    @abstractmethod
    def getReward(self,state):
        """
        given the manipulator state determine the reward
        """
        pass

    @abstractmethod
    def getTerminal(self,state):
        """
        given the manipulator state determine if the state is terminal
        """
        pass

    def extraSteps(self):
        """
        in case the environment has to do extra steps after the dynamics are done
        for example:
            if a trajectory is being tracked, the trajectory may need to be stepped along
        default is do nothing
        """
        pass

class DynamicReaching(EnvironmentWrapper):
    """
    For dynamic reaching there is a single target that is repeatedly trying to be reached
    should start from the reference configuration every time
    can change between only tip state and tip and mid state
    """

    def __init__(self, manipulator, target, tipOnly=True, bound = 30/100, terminal = 2/100):
        self.target = target #the target goal
        self.tipOnly = tipOnly
        self.bound = bound #the boundary percentage
        self.terminal = terminal #the terminal percentage

        #defining observation_space
        angle_low = [-np.pi]*3
        angle_high = [np.pi]*3
        pos_low = [-0.1,-0.1,0.035]
        pos_high = [0.1,0.1,0.045]

        low_vel = [-10]*6
        high_vel = [10]*6

        if self.tipOnly:
            observation_space = spaces.Box(low = np.array(angle_low+pos_low+low_vel), high = np.array(angle_high+pos_high+high_vel), dtype=np.float32)
        else: #return tip and mid
            observation_space = spaces.Box(low = np.array((angle_low+pos_low+low_vel)*2), high = np.array((angle_high+pos_high+high_vel)*2), dtype=np.float32)

        super().__init__(manipulator,observation_space)


    def getGoal(self):
        return self.target

    def envReset(self):
        pass #there is nothign to do

    def manipulatorReset(self):
        state = self.manipulator.initState()
        self.manipulator.eng.close("all",nargout=0) #close drawings if they are there
        return state

    def getObservation(self,state):
        #pick out the relevant states
        g = state['g']
        eta = state['eta']
        if self.tipOnly:
            R = np.array([g[-1,0:3],g[-1,3:6],g[-1,6:9]])
            angles = self.manipulator.eng.extractAngles(np2mat(R))
            angles = mat2np(angles).T[0]
            return np.hstack([angles,g[-1,9:12],eta[-1,:]])
        else:
            R_tip = np.array([g[-1,0:3],g[-1,3:6],g[-1,6:9]])
            angles_tip = self.manipulator.eng.extractAngles(np2mat(R_tip))
            angles_tip = mat2np(angles_tip).T[0]
            state_tip = np.hstack([angles_tip,g[-1,9:12],eta[-1,:]])

            mid = self.manipulator.n//2 + 1 #might not actually be the middle
            R_mid = np.array([g[mid,0:3],g[mid,3:6],g[mid,6:9]])
            angles_mid = self.manipulator.eng.extractAngles(np2mat(R_mid))
            angles_mid = mat2np(angles_mid).T[0]
            state_mid = np.hstack([angles_mid,g[mid,9:12],eta[mid,:]])
            return np.hstack([state_tip,state_mid])

    def getReward(self,state):
        #the reward for this case is simply the distance between the tip position and the goal
        g = state['g']
        tip = g[-1,9:12]
        dist = np.linalg.norm(tip-self.getGoal())
        #return self.rewardScale*dist
        bound = 0.04*self.bound
        if dist <= bound:
            return -dist/bound
        else:
            return -1

    def getTerminal(self,state):
        #since this should be a point just outside the workspace, do not want to keep running once the point is reached
        g = state['g']
        tip = g[-1,9:12]
        dist = np.linalg.norm(tip-self.getGoal())

        return dist<0.04*self.terminal #withing _ percent tip error
        #return False


class VariableTarget(EnvironmentWrapper):
    """
    This type of environment will vary the target goal randomly
    """

    def __init__(self,manipulator, workspace, tipOnly=True, bound = 30/100, terminal = 2/100):
        self.workspace = workspace #workspace to sample goal from
        self.tipOnly = tipOnly
        self.bound= bound
        self.terminal = terminal

        #defining observation_space
        angle_low = [-np.pi]*3
        angle_high = [np.pi]*3
        pos_low = [-0.1,-0.1,0.035]
        pos_high = [0.1,0.1,0.045]

        low_vel = [-10]*6
        high_vel = [10]*6

        #include the goal into the observation space

        if self.tipOnly:
            observation_space = spaces.Box(low = np.array(pos_low + angle_low+pos_low+low_vel), high = np.array(pos_high + angle_high+pos_high+high_vel), dtype=np.float32)
        else: #return tip and mid
            observation_space = spaces.Box(low = np.array(pos_low + (angle_low+pos_low+low_vel)*2), high = np.array(pos_high + (angle_high+pos_high+high_vel)*2), dtype=np.float32)

        super().__init__(manipulator,observation_space)

        self.goal = self.genGoal()

    def getGoal(self):
        return self.goal

    def envReset(self):
        #reset the goal
        self.goal = self.genGoal()

    def manipulatorReset(self):
        state = self.manipulator.initState() #might as well just start from the initial reference state, but this is a candidate for not resetting
        self.manipulator.eng.close("all",nargout=0) #close drawings if they are there
        return state

    def getObservation(self,state):
        #pick out the relevant states
        g = state['g']
        eta = state['eta']

        if self.tipOnly:
            R = np.array([g[-1,0:3],g[-1,3:6],g[-1,6:9]])
            angles = self.manipulator.eng.extractAngles(np2mat(R))
            angles = mat2np(angles).T[0]
            obs = np.hstack([angles,g[-1,9:12],eta[-1,:]])
        else:
            R_tip = np.array([g[-1,0:3],g[-1,3:6],g[-1,6:9]])
            angles_tip = self.manipulator.eng.extractAngles(np2mat(R_tip))
            angles_tip = mat2np(angles_tip).T[0]
            state_tip = np.hstack([angles_tip,g[-1,9:12],eta[-1,:]])

            mid = self.manipulator.n//2 + 1 #might not actually be the middle
            R_mid = np.array([g[mid,0:3],g[mid,3:6],g[mid,6:9]])
            angles_mid = self.manipulator.eng.extractAngles(np2mat(R_mid))
            angles_mid = mat2np(angles_mid).T[0]
            state_mid = np.hstack([angles_mid,g[mid,9:12],eta[mid,:]])
            obs = np.hstack([state_tip,state_mid])

        obs = np.hstack([self.getGoal(),obs])
        return obs

    def getReward(self,state):
        #the reward for this case is simply the distance between the tip position and the goal
        g = state['g']
        tip = g[-1,9:12]
        dist = np.linalg.norm(tip-self.getGoal())
        bound = 0.04*self.bound
        if dist <= bound:
            return -dist/bound
        else:
            return -1
        #return self.rewardScale*dist

    def getTerminal(self,state):
        #this is a random point in the workspace, so given that some are only dynamically reachable it should terminate
        g = state['g']
        tip = g[-1,9:12]
        dist = np.linalg.norm(tip-self.getGoal())


        return dist<0.04*self.terminal

    def genGoal(self):
        goal = self.workspace.sample()
        return goal

    def setGoal(self,g):
        self.goal = g



class VariableTrajectory(EnvironmentWrapper):
    """
    This type of environment will vary the target goal as a trajectory
    may be useful to augment the workspace a bit to gaurantee that the trajectory is in a reasonable location
    """

    def __init__(self,manipulator, workspace, tau, tipOnly=True, bound = 30/100):
        self.manipulator = manipulator #I thought this would occur in the call to super()
        self.workspace = workspace #workspace to sample goal from
        self.tipOnly = tipOnly
        self.bound = bound

        self.manualGoal = False #be able to manually set the goal

        #defining observation_space
        #confusing naming scheme, whoops
        angle_low = [-np.pi]*3
        angle_high = [np.pi]*3
        pos_low = [-0.1,-0.1,0.035]
        pos_high = [0.1,0.1,0.045]

        low_vel = [-10]*6
        high_vel = [10]*6

        #include the goal into the observation space, the goal is a position and velocity

        if self.tipOnly:
            observation_space = spaces.Box(low = np.array(pos_low + low_vel[:3] + angle_low+pos_low+low_vel), high = np.array(pos_high + high_vel[:3] + angle_high+pos_high+high_vel), dtype=np.float32)
        else: #return tip and mid
            observation_space = spaces.Box(low = np.array(pos_low + low_vel[:3] + (angle_low+pos_low+low_vel)*2), high = np.array(pos_high + high_vel[:3] + (angle_high+pos_high+high_vel)*2), dtype=np.float32)

        #the parameters for the trajectory definition
        self.tau = tau
        self.steps = 0
        self.traj = self.genTrajectory() #generate the dicitionary defining the properties in the trajectory

        self.goal = self.getGoal()

        super().__init__(manipulator,observation_space)



    def extraSteps(self):
        #need to step forward the goal trajectory after every dynamics step
        self.steps += 1

    def getGoal(self):
        if self.manualGoal:
            return self.goal
        else:
            #compute the poisiton along the trjectory
            p1 = self.traj['p1']
            p2 = self.traj['p2']
            v1 = self.traj['v1']
            v2 = self.traj['v2']

            alpha0 = p1
            alpha1 = v1
            alpha2 = -(3*p1-3*p2+self.tau*(2*v1+v2))/self.tau**2
            alpha3 = (2*p1-2*p2+self.tau*(v1+v2))/self.tau**3

            t = self.steps*self.manipulator.dt

            g_pos = alpha0+alpha1*t+alpha2*t**2+alpha3*t**3
            g_vel = alpha1+2*alpha2*t+3*alpha3*t**2

            self.goal = np.concatenate([g_pos,g_vel])

            return self.goal

    def envReset(self):
        #reset the goal
        self.traj = self.genTrajectory()
        self.steps = 0

    def manipulatorReset(self):
        state = self.manipulator.initState() #might as well just start from the initial reference state, but this is a candidate for not resetting
        self.manipulator.eng.close("all",nargout=0) #close drawings if they are there
        return state

    def getObservation(self,state):
        #pick out the relevant states
        g = state['g']
        eta = state['eta']

        if self.tipOnly:
            R = np.array([g[-1,0:3],g[-1,3:6],g[-1,6:9]])
            angles = self.manipulator.eng.extractAngles(np2mat(R))
            angles = mat2np(angles).T[0]
            obs = np.hstack([angles,g[-1,9:12],eta[-1,:]])
        else:
            R_tip = np.array([g[-1,0:3],g[-1,3:6],g[-1,6:9]])
            angles_tip = self.manipulator.eng.extractAngles(np2mat(R_tip))
            angles_tip = mat2np(angles_tip).T[0]
            state_tip = np.hstack([angles_tip,g[-1,9:12],eta[-1,:]])

            mid = self.manipulator.n//2 + 1 #might not actually be the middle
            R_mid = np.array([g[mid,0:3],g[mid,3:6],g[mid,6:9]])
            angles_mid = self.manipulator.eng.extractAngles(np2mat(R_mid))
            angles_mid = mat2np(angles_mid).T[0]
            state_mid = np.hstack([angles_mid,g[mid,9:12],eta[mid,:]])
            obs = np.hstack([state_tip,state_mid])

        obs = np.hstack([self.getGoal(),obs])
        return obs

    def getReward(self,state):
        #still only rewarding a similar position
        g = state['g']
        tip = g[-1,9:12]
        dist = np.linalg.norm(tip-self.getGoal()[:3])
        bound = 0.04*self.bound

        if dist<=bound:
            return -dist/bound
        else:
            return -1

        #return self.rewardScale*dist

    def getTerminal(self,state):
        #no good terminal state, so never terminal
        return False

    def genTrajectory(self):
        #need to generate positions and velocities to interpolate

        p1 = self.workspace.sample()
        p2 = self.workspace.sample()

        #probably could do better
        v1 = np.random.uniform(0,0.01,3)
        v2 = np.random.uniform(0,0.01,3)

        return {'p1':p1, 'p2':p2, 'v1':v1, 'v2':v2}

    def setGoal(self,g):
        self.goal = g #mostly useful for plotting
