from keras.callbacks import Callback
import numpy as np


class PolicyEval(Callback):
    def __init__(self, eval_func, record=10):
        # want to have an evaluation function that takes in the policy and gives a measure of the performance
        self.eval_func = eval_func
        self.perfs = []
        self.record = record
    def on_epoch_end(self,epoch,logs=None):
        if epoch%self.record == 0: #only store every 10 epochs
            actor = self.model
            res = self.eval_func(actor)
            self.perfs.append(res)

def performanceDynamicReaching(env, actor):
    #for this it is pretty straightforward
    #see if the tip position can be reached
    #and see how many steps it took?
    #for now just return 1 if it achieved the reach, 0 if it did not
    #changin my mind, just make it accumulated reward

    steps = 100
    obs = env.reset()
    val = 0
    for _ in range(steps):
        a = actor.forward(obs)
        obs, reward, terminal, _ = env.step(a)
        val+=reward
        if terminal:
            return val

    return val

def performanceVariableTarget(env, actor, r, z):
    #the goal here is to have a consistent goal to evaluate the policy on to track progress
    #for the variable target, want to see if a selection of points can be tracked to
    #for this it is the average final tip error
    #may want to get points from the workspace
    #this definately needs to be changed to be points from the workspace
    pts = [np.array([r*np.cos(i*np.pi/2),r*np.sin(i*np.pi/2),z]) for i in range(4)]
    steps = 50
    perf = []
    for pt in pts:
        obs = env.reset()
        obs[:3] = pt
        env.setGoal(pt)
        for _ in range(steps):
            a = actor.forward(obs)
            obs, reward, terminal , _ = env.step(a)
            if terminal:
                break
            #env.render()
        pos_err = np.linalg.norm(obs[:3]-obs[6:9]) #the last tip error
        perf.append(pos_err/0.04) #normalized by length
    return np.mean(perf)

def performanceVariableTrajectory(env, actor, z, r, steps):
    #for this it is probably reasonable to test on the tracking of a circle
    w = np.pi*2/steps
    circle = lambda t: np.array([r*np.cos(w*t), r*np.sin(w*t), z])
    circle_vel = lambda t: np.array([-w*r*np.sin(w*t), w*r*np.cos(w*t), 0])
    env.manualGoal = True


    tip_err = []
    obs = env.reset()
    for t in range(steps):
        goal_pos = circle(t)
        goal_vel = circle_vel(t)
        goal = np.concatenate([goal_pos,goal_vel])
        env.setGoal(goal)
        obs[:6] = goal

        a = actor.forward(obs)
        obs, _ ,_ , _ = env.step(a)
        #env.render()
        tip_err.append(np.linalg.norm(goal[:3]-obs[6:9])/0.04)
    env.manualGoal = False
    return np.mean(tip_err)
