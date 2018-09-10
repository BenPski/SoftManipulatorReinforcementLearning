"""

Sketching out how the individual runs will work for the reinforcement learning experiments

Every experiment will use the same DDPG structure
each experiment changes based off of the task being learned
    for the tasks the manipulator changes, the environemnt changes, and the performance changes

will go like this
    Dynamic Reaching:
        each 100000 samples
        6 different targets (need to determine the 6 targets for cable and the tca seperately)
            env = cable, tip, target
            perf = cable dynReaching

            env = cable, both, target
            perf = cable dynReaching

            env = tca, tip , target
            per = tca dynReaching

            env = tca, both, target
            perf = tca dynreaching

    variable target (cable and tca have different workspaces to select from )
        repeat 5 times
            each 200000 samples

            env = cablae, tip
            perf = cable vartarg

            env = cable, both
            perf = cable vcartarg

            env = tca, tip
            perf = tca vartarg

            env = tca, both
            perf = tca vartarg

    variable trajectory (cable and tca have different workspaces to select from)
        repeat 5 times
            each 500000 samples
                env = cable, tip
                perf = cable vartraj

                env = cable, both
                perf = cable vartraj

                env = tca, tip
                perf = tca vartraj

                env = tca, both
                perf = tca vartraj


"""

import Manipulators as M #contains the manipulator dynamics definitions
import Workspace as W #the workspace
import EnvironmentWrappers as EW #the wrappers to take manipulators to environments
import Performance as P #performance measures
import numpy as np
import pickle
import os.path

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
import rl.random

from manipAndWorkspace import manips, workspaces
from network_utils import generateAgent

import argparse
import configparser
import uuid
import datetime

class Runner(object):
    """
    runs a test and saves configs
    """

    def __init__(self,manip,state,task,samples,steps=100,actor_hidden=[100,100],actor_act='relu',critic_hidden=[100,100],critic_act='relu'):

        #get things associated properly with the manipulator
        self.manip_name = manip
        self.manip = manips[manip]
        self.static_workspace = workspaces[manip+'_static']
        self.dynamic_workspace = workspaces[manip+'_dynamic']

        #the state to be measured
        self.state = state
        self.tip = state == 'tip'

        #setup environment
        self.task = task
        if task=='dynReach':
            self.target = getTargets(self.static_workspace, self.dynamic_workspace,1)[0]
            self.env = EW.DynamicReaching(self.manip, self.target, tipOnly = self.tip)
        elif task=='varTarg':
            self.env = EW.VariableTarget(self.manip, self.dynamic_workspace, tipOnly = tip)
        elif task=='varTraj':
            self.tau = steps*self.manip.dt
            self.env = EW.VariableTrajectory(self.manip, self.static_workspace, self.tau, tipOnly = tip)

        #setup the ddpg agent and networks
        self.samples = samples
        self.steps = steps
        self.actor_hidden = actor_hidden
        self.actor_act = actor_act
        self.critic_hidden = critic_hidden
        self.critic_act = critic_act

        self.agent = generateAgent(self.env,self.actor_hidden,self.actor_act,self.critic_hidden,self.critic_act)

        #setup config
        #for the config need to know manipulator information
        self.config = configparser.ConfigParser()
        self.config['Manipulator'] = {'manip':self.manip_name}
        self.config['State'] = {'state': self.state}
        if self.task == 'dynReach':
            self.config['Task'] = {'task':self.task, 'target': self.target}
        elif self.task == 'varTarg':
            self.config['Task'] = {'task':self.task}
        elif self.task == 'varTraj':
            self.config['Task'] = {'task':self.task, 'tau':self.tau}

        self.config['Test'] = {'perf_location': ""} #no performance file exists yet

        self.config_base_location = "./tests/"+self.manip_name+'_'+self.state+'_'+self.task
        self.weights_base_location = "./weights/"+self.manip_name+'_'+self.state+'_'+self.task

    def run(self):
        #simply run the fitting process
        self.agent.fit(self.env, nb_steps=self.samples, visualize=False, verbose=2, nb_max_episode_steps=self.steps)

        #then save the results
        self.save()


    def save(self):
        #want to save the config and have it point to the relevant weights location
        #make sure the relevant directories exist
        try:
            os.makedirs('tests')
        except:
            pass #likely already exists, so no need to worry
        try:
            os.makedirs('weights')
        except:
            pass

        weights_name = self.weights_base_location + '_' + str(uuid.uuid4()) + '.h5f' #append a rnadom uuid to the end of the name so that it is unique
        self.agent.save_weights(weights_name,overwrite=False) #should never need to overwrite
        #now that the weights are saved write down the file location
        self.config['Networks'] = {'weights_location':weights_name, 'actor_hidden':self.actor_hidden, 'actor_act':self.actor_act, 'critic_hidden':self.critic_hidden, 'critic_act': self.critic_act}
        #for saving the config we want a name that makes some sense so that it is simple to keep track of
        #append the timestamp to the name
        config_name = self.config_base_location + '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.ini'
        with open(config_name,'w') as f:
            self.config.write(f)
        print("Config saved to: ", config_name)







def run(agent, samples, steps, env, performance, weights_file, perf_file):
    """
    train the agent on the environment and record the performance and the output
    """

    try: #don't repeat a test
        agent.load_weights(weights_file)
    except:
        agent.fit(env, nb_steps=samples, visualize=False, verbose=2, nb_max_episode_steps=steps)#, callbacks = [performance]), don't feel like having performance

        agent.save_weights(weights_file, overwrite=True)

    if os.path.isfile(perf_file): #don't need to resave performance
        with open(perf_file,'rb') as f:
            performance.perfs = pickle.load(f)
    else:
        with open(perf_file, 'wb') as f:
            pickle.dump(performance.perfs, f)

    return (agent, performance.perfs)

def runDynamicReaching(samples, steps, manip, target, tip, weights_file, perf_file, bound):
    """
    run the dynamic reaching
    """

    env = EW.DynamicReaching(manip, target, tipOnly = tip, bound=bound)
    perf = P.PolicyEval(lambda actor: P.performanceDynamicReaching(env,actor))

    nb_actions = env.action_space.shape[0]

    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(30))
    actor.add(Activation('relu'))
    actor.add(Dense(30))
    actor.add(Activation('relu'))
    actor.add(Dense(30))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('sigmoid')) # get output to go from 0 to 1

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = merge([action_input, flattened_observation], mode='concat')
    x = Dense(32)(x)
    x = Activation('tanh')(x)
    x = Dense(32)(x)
    x = Activation('tanh')(x)
    x = Dense(32)(x)
    x = Activation('tanh')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(input=[action_input, observation_input], output=x)

    memory = SequentialMemory(limit=500000, window_length=1)
    random_process = rl.random.OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=0.99, target_model_update=1e-3)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    return run(agent, samples, steps, env, perf, weights_file, perf_file)

def runVariableTarget(samples, steps, manip, workspace, tip, weights_file, perf_file):
    """
    run the variable target task
    """

    (r,z) = getVariableTargets(workspace)

    env = EW.VariableTarget(manip, workspace, tipOnly = tip)
    perf = P.PolicyEval(lambda actor: P.performanceVariableTarget(env, actor, r, z))

    nb_actions = env.action_space.shape[0]

    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(30))
    actor.add(Activation('relu'))
    actor.add(Dense(30))
    actor.add(Activation('relu'))
    actor.add(Dense(30))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('sigmoid')) # get output to go from 0 to 1

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = merge([action_input, flattened_observation], mode='concat')
    x = Dense(32)(x)
    x = Activation('tanh')(x)
    x = Dense(32)(x)
    x = Activation('tanh')(x)
    x = Dense(32)(x)
    x = Activation('tanh')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(input=[action_input, observation_input], output=x)

    memory = SequentialMemory(limit=500000, window_length=1)
    random_process = rl.random.OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=0.99, target_model_update=1e-3)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    return run(agent, samples, steps, env, perf, weights_file, perf_file)

def runVariableTrajectory(samples, steps, manip, workspace, tip, z, r, steps_perf, weights_file, perf_file):
    """
    run the variable trajectory task
    """

    env = EW.VariableTrajectory(manip, workspace, steps*manip.dt, tipOnly = tip)
    perf = P.PolicyEval(lambda actor: P.performanceVariableTrajectory(env, actor, z, r, steps_perf))

    nb_actions = env.action_space.shape[0]

    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(30))
    actor.add(Activation('relu'))
    actor.add(Dense(30))
    actor.add(Activation('relu'))
    actor.add(Dense(30))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('sigmoid')) # get output to go from 0 to 1

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = merge([action_input, flattened_observation], mode='concat')
    x = Dense(32)(x)
    x = Activation('tanh')(x)
    x = Dense(32)(x)
    x = Activation('tanh')(x)
    x = Dense(32)(x)
    x = Activation('tanh')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(input=[action_input, observation_input], output=x)

    memory = SequentialMemory(limit=500000, window_length=1)
    random_process = rl.random.OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=0.99, target_model_update=1e-3)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])


    return run(agent, samples, steps, env, perf, weights_file, perf_file)

def DynamicReaching():
    """
    the dynamic reaching routine
    """

    samples = 20000
    steps = 100

    #first do the cables
    if os.path.exists('cable_targets.pkl'): #already generated targets
        with open('cable_targets.pkl','rb') as f:
            targets = pickle.load(f)
    else:
        targets = getTargets(staticWorkspaceCable, dynamicWorkspaceCable, 3)
        with open('cable_targets.pkl','wb') as f:
            pickle.dump(targets, f)

    bound = staticWorkspaceCable.max_r
    print(bound)

    print("\tStarting Cable")
    print("\t targets: ", targets)
    #tip
    for (i,target) in enumerate(targets):
        print("\t\tTip iteration: ", i)
        runDynamicReaching(samples, steps, manipCable, target, True, "cable_DynReach_tip_weights_" + str(i) + ".h5f", "cable_DynReach_tip_perf_" + str(i) + ".pkl",bound)

    #now do with both the states
    for (i,target) in enumerate(targets):
        print("\t\tBoth iteration: ", i)
        runDynamicReaching(samples, steps, manipCable, target, False, "cable_DynReach_both_weights_" + str(i) + ".h5f", "cable_DynReach_both_perf_" + str(i) + ".pkl",bound)

    #do the tca
    if os.path.exists('tca_targets.pkl'): #already generated targets
        with open('tca_targets.pkl','rb') as f:
            targets = pickle.load(f)
    else:
        targets = getTargets(staticWorkspaceTCA, dynamicWorkspaceTCA, 3)
        with open('tca_targets.pkl','wb') as f:
            pickle.dump(targets, f)

    bound = staticWorkspaceTCA.max_r
    print(bound)

    print("\tStarting TCA")
    print("\t targets: ", targets)
    #tip
    for (i,target) in enumerate(targets):
        print("\t\tTip iteration: ", i)
        runDynamicReaching(samples, steps, manipTCA, target, True, "tca_DynReach_tip_weights_" + str(i) + ".h5f", "tca_DynReach_tip_perf_" + str(i) + ".pkl",bound)

    #now do with both the states
    for (i,target) in enumerate(targets):
        print("\t\tBoth iteration: ", i)
        runDynamicReaching(samples, steps, manipTCA, target, False, "tca_DynReach_both_weights_" + str(i) + ".h5f", "tca_DynReach_both_perf_" + str(i) + ".pkl",bound)

def VariableTarget():
    """
    the variable target routine
    """

    samples = 200000
    steps = 100

    REPLICATES = 3

    print("\tStarting Cable")
    #do the cable first
    #tip
    for i in range(REPLICATES):
        print("\t\tTip replicate: ", i)
        runVariableTarget(samples, steps, manipCable, dynamicWorkspaceCable, True, "cable_VarTarg_tip_weights_" + str(i) + ".h5f", "cable_VarTarg_tip_perf_" + str(i) + ".pkl")

    #both
    for i in range(REPLICATES):
        print("\t\tBoth replicate: ", i)
        runVariableTarget(samples, steps, manipCable, dynamicWorkspaceCable, False, "cable_VarTarg_both_weights_" + str(i) + ".h5f", "cable_VarTarg_both_perf_" + str(i) + ".pkl")

    #now tcas
    print("\tStarting TCA")
    #tip
    for i in range(REPLICATES):
        print("\t\tTip replicate: ", i)
        runVariableTarget(samples, steps, manipTCA, dynamicWorkspaceTCA, True, "tca_VarTarg_tip_weights_" + str(i) + ".h5f", "tca_VarTarg_tip_perf_" + str(i) + ".pkl")

    #both
    for i in range(REPLICATES):
        print("\t\tBoth replicate: ", i)
        runVariableTarget(samples, steps, manipCable, dynamicWorkspaceTCA, False, "tca_VarTarg_both_weights_" + str(i) + ".h5f", "tca_VarTarg_both_perf_" + str(i) + ".pkl")


def VariableTrajectory():
    """
    the variable trajectory routine
    """
    samples = 500000
    steps = 100

    REPLICATES = 3

    #do the cable first
    if os.path.exists('cable_trajectory_params.pkl'): #already generated
        with open('cable_trajectory_params.pkl','rb') as f:
            z,r = pickle.dump(f)
    else:
        z,r = getCircleTrajectory(staticWorkspaceCable)
        with open('cable_trajectory_params.pkl','wb') as f:
            pickle.dump((z,r), f)

    print("\tStarting Cable")
    print("\tValues: (z,r) ", z, r)

    #tip
    for i in range(REPLICATES):
        print("\t\tTip replicate: ", i)
        runVariableTrajectory(samples, steps, manipCable, staticWorkspaceCable, True, z, r, steps, "cable_VarTraj_tip_weights_" + str(i) + ".h5f", "cable_VarTraj_tip_perf_" + str(i) + ".pkl")

    #both
    for i in range(REPLICATES):
        print("\t\tBoth replicate: ", i)
        runVariableTrajectory(samples, steps, manipCable, staticWorkspaceCable, False, z, r, steps, "cable_VarTraj_both_weights_" + str(i) + ".h5f", "cable_VarTraj_both_perf_" + str(i) + ".pkl")

    #now tcas
    if os.path.exists('tca_trajectory_params.pkl'): #already generated
        with open('tca_trajectory_params.pkl','rb') as f:
            z,r = pickle.dump(f)
    else:
        z,r = getCircleTrajectory(staticWorkspaceTCA)
        with open('tca_trajectory_params.pkl','wb') as f:
            pickle.dump((z,r), f)

    print("\tStarting TCA")
    print("\tValues: (z,r) ", z, r)
    #tip
    for i in range(REPLICATES):
        print("\t\tTip replicate: ", i)
        runVariableTrajectory(samples, steps, manipTCA, staticWorkspaceCTCA, True, z, r, steps, "tca_VarTraj_tip_weights_" + str(i) + ".h5f", "tca_VarTraj_tip_perf_" + str(i) + ".pkl")

    #both
    for i in range(REPLICATES):
        print("\t\tBoth replicate: ", i)
        runVariableTrajectory(samples, steps, manipTCA, staticWorkspaceTCA, False, z, r, steps, "tca_VarTraj_both_weights_" + str(i) + ".h5f", "tca_VarTraj_both_perf_" + str(i) + ".pkl")


def getTargets(staticWorkspace, dynamicWorkspace, n, perturb=1e-3):
    """
    need to get the dynamic reaching targets from the static Workspace

    one way is to pullout the vertices of the workspace and perturb them all slightly and keep the ones that are outside the workspace
    """

    vertices = staticWorkspace.vertices

    targets = []
    while len(targets) < n:
        i = np.random.randint(0, vertices.shape[0])

        vert = vertices[i,:]

        point = vert + np.random.uniform(-1*np.array(3*[perturb]), np.array(3*[perturb]))

        if (not staticWorkspace.inside(point)) and dynamicWorkspace.inside(point):
            targets.append(point)

    return targets


def getVariableTargets(workspace):
    #generate r and z values for the workspace
    r = workspace.max_r
    z = (workspace.max_z+workspace.min_z)/2

    return (r*0.9,z)

def getCircleTrajectory(staticWorkspace):
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


def runAll():
    print("Running Dynamic Reaching")
    DynamicReaching()
    print("Running Variable Target")
    VariableTarget()
    print("Running Variable Trajectory")
    VariableTrajectory()


if __name__ == "__main__":
    #runAll()
    parser = argparse.ArgumentParser(description="Running a learning routine")
    parser.add_argument('manipulator', choices = ['cable','tca'], metavar='manip', type=str, help='The type of manipulator')
    parser.add_argument('state', choices=['tip','both'], metavar='state', type=str, help='The measured state')
    parser.add_argument('task', choices=['dynReach', 'varTarg', 'varTraj'], metavar='task', type=str, help='The task to be trained on')
    parser.add_argument('samples', metavar='samples', type=int, help='The number of samples to train on')

    parser.add_argument('--actor_hidden', metavar='N', type=int, nargs='+', default=[100,100], help='The hidden network structure for the actor')
    parser.add_argument('--actor_activation', metavar='act', type=str, choices=['relu','tanh'], default='relu', help='The activation function for the actor')
    parser.add_argument('--critic_hidden', metavar='N', type=int, nargs='+', default=[100,100], help='The hidden network structure for the critic')
    parser.add_argument('--critic_activation', metavar='act', type=str, choices=['relu','tanh'], default='relu', help='The activation function for the critic')

    args = parser.parse_args()
    #print(args)

    #run the training given the arguments

    test = Runner(args.manipulator,args.state,args.task,args.samples,actor_hidden=args.actor_hidden,actor_act=args.actor_activation,critic_hidden=args.critic_hidden,critic_act=args.critic_activation)
    test.run()
