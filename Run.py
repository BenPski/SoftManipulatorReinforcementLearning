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
#for some reason have to import performance first, or else there is an error with loading tensorflow
import Performance as P #performance measures
import Manipulators as M #contains the manipulator dynamics definitions
import Workspace as W #the workspace
import EnvironmentWrappers as EW #the wrappers to take manipulators to environments
import numpy as np
import pickle
import os.path

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
import rl.random

manipCable = M.CableManipulator(10,0.01,0.1)

manipTCA = M.TCAManipulator(5,0.1,3)

#the workspace file locations
def getStaticWorkspace(manip,filename, **kwargs):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        workspace = W.makeStaticManipulatorWorkspace(manip, **kwargs)
        with open(filename, 'wb') as f:
            pickle.dump(workspace, f)
        return workspace

def getDynamicWorkspace(manip, staticWorkspace, filename, **kwargs):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        #workspace = W.makeDynamicManipulatorWorkspace(manip, **kwargs)
        workspace = W.dynamicWorkspace(manip, staticWorkspace, **kwargs)
        with open(filename, 'wb') as f:
            pickle.dump(workspace, f)
        return workspace

cable_static_workspace = "cable_static_workspace.pkl"
tca_static_workspace = "tca_static_workspace.pkl"

cable_dynamic_workspace = "cable_dynamic_workspace.pkl"
tca_dynamic_workspace = "tca_dynamic_workspace.pkl"

#only need to use the tip state
print("loading workspaces")
staticWorkspaceCable = getStaticWorkspace(manipCable,cable_static_workspace)
print("got static cable")
staticWorkspaceTCA = getStaticWorkspace(manipTCA,tca_static_workspace, STEPS = 200)
print("got static tca")
dynamicWorkspaceCable = getDynamicWorkspace(manipCable,staticWorkspaceCable,cable_dynamic_workspace)
print("got dynamic cable")
dynamicWorkspaceTCA = getDynamicWorkspace(manipTCA,staticWorkspaceTCA,tca_dynamic_workspace, SAMPLES = 50, STEPS = 100)
print("got dynamic tca")

#dynamicWorkspaceTCA.plot(), # I still don't like the workspace, but it is getting annoying at this point


def run(agent, samples, steps, env, performance, weights_file, perf_file):
    """
    train the agent on the environment and record the performance and the output
    """

    try: #don't repeat a test
        agent.load_weights(weights_file)
    except:
        agent.fit(env, nb_steps=samples, visualize=False, verbose=2, nb_max_episode_steps=steps, callbacks = [performance])

        agent.save_weights(weights_file, overwrite=True)

    if os.path.isfile(perf_file): #don't need to resave performance
        with open(perf_file,'rb') as f:
            performance.perfs = pickle.load(f)
    else:
        with open(perf_file, 'wb') as f:
            pickle.dump(performance.perfs, f)

    return (agent, performance.perfs)

def runDynamicReaching(samples, steps, manip, target, tip, weights_file, perf_file):
    """
    run the dynamic reaching
    """

    env = EW.DynamicReaching(manip, target, tipOnly = tip)
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
    x = Concatenate(axis=-1)([action_input, flattened_observation])
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
    x = Concatenate(axis=-1)([action_input, flattened_observation])
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
    x = Concatenate(axis=-1)([action_input, flattened_observation])
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

    samples = 10000
    steps = 100

    #first do the cables
    targets = getTargets(staticWorkspaceCable, 3)
    with open('cable_targets.pkl','wb') as f:
        pickle.dump(targets, f)

    print("\tStarting Cable")
    print("\t targets: ", targets)
    #tip
    for (i,target) in enumerate(targets):
        print("\t\tTip iteration: ", i)
        runDynamicReaching(samples, steps, manipCable, target, True, "cable_DynReach_tip_weights_" + str(i) + ".h5f", "cable_DynReach_tip_perf_" + str(i) + ".pkl")

    #now do with both the states
    for (i,target) in enumerate(targets):
        print("\t\tBoth iteration: ", i)
        runDynamicReaching(samples, steps, manipCable, target, False, "cable_DynReach_both_weights_" + str(i) + ".h5f", "cable_DynReach_both_perf_" + str(i) + ".pkl")

    #do the tca
    targets = getTargets(staticWorkspaceTCA, 3)
    with open('tca_targets.pkl','wb') as f:
        pickle.dump(targets,f)

    print("\tStarting TCA")
    print("\t targets: ", targets)
    #tip
    for (i,target) in enumerate(targets):
        print("\t\tTip iteration: ", i)
        runDynamicReaching(samples, steps, manipTCA, target, True, "tca_DynReach_tip_weights_" + str(i) + ".h5f", "tca_DynReach_tip_perf_" + str(i) + ".pkl")

    #now do with both the states
    for (i,target) in enumerate(targets):
        print("\t\tBoth iteration: ", i)
        runDynamicReaching(samples, steps, manipTCA, target, False, "tca_DynReach_both_weights_" + str(i) + ".h5f", "tca_DynReach_both_perf_" + str(i) + ".pkl")

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
    z,r = getCircleTrajectory(staticWorkspaceCable)
    print("\tStarting TCA")
    print("\tValues: (z,r) ", z, r)

    with open('cable_trajectory_params.pkl','wb') as f:
        pickle.dump((z,r), f)
    #tip
    for i in range(REPLICATES):
        print("\t\tTip replicate: ", i)
        runVariableTrajectory(samples, steps, manipTCA, staticWorkspaceCTCA, True, z, r, steps, "tca_VarTraj_tip_weights_" + str(i) + ".h5f", "tca_VarTraj_tip_perf_" + str(i) + ".pkl")

    #both
    for i in range(REPLICATES):
        print("\t\tBoth replicate: ", i)
        runVariableTrajectory(samples, steps, manipTCA, staticWorkspaceTCA, False, z, r, steps, "tca_VarTraj_both_weights_" + str(i) + ".h5f", "tca_VarTraj_both_perf_" + str(i) + ".pkl")


def getTargets(staticWorkspace, n, perturb=1e-3):
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

        if staticWorkspace.inside(point):
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
    runAll()
