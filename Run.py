"""
Controls the running of a reinforcement learning process
"""

from config_utils import ConfigHandler
import argparse


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
    #run the training given the arguments
    
    test = ConfigHandler.setupNew(args.manipulator,args.state,args.task, {'samples':args.samples, 'steps':100}, {'actor_hidden':args.actor_hidden,'actor_act':args.actor_activation,'critic_hidden':args.critic_hidden,'critic_act':args.critic_activation})
    test.run_train(save=True)
