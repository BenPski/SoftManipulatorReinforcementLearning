# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 08:33:51 2018

@author: benpski



A new version of the Run script
Essentially just made to interface better with the new config/controller setup

"""
from ConfigHandler import ConfigHandler
import argparse


if __name__ == "__main__":
    
   #the main parser only takes the manipulator
   #the subparsers take the information for the task, the only reason for this way is for aesthetic purposes
   #many parameters overlap, but it makes more sense to specify the task parameters after stating the task
    
    #will always just have the default network structure for now
    
        
    parser = argparse.ArgumentParser(description="Running a learning routine")
    subparsers = parser.add_subparsers(title='Tasks', description='The possible tasks to train on.', help='The different tasks.')
    
    #manip
    parser.add_argument('manipulator', choices = ['cable','tca'], type=str, help='The type of manipulator')
    
    #dynamic reaching
    parser_dynReach = subparsers.add_parser('dynReach', help='The dynamic reaching task.')
    parser_dynReach.add_argument('measure', type=int, help='The number of states to observe.')
    parser_dynReach.add_argument('train_steps', type=int, help='The number of steps in a training episode.')
    parser_dynReach.add_argument('train_samples', type=int, help='The total number of samples to train on.')
    parser_dynReach.add_argument('train_bound', type=float, help='The coefficient for the reward measurement.')
    parser_dynReach.add_argument('train_terminal', type=float, help='The terminal measure for the training.')
    parser_dynReach.add_argument('test_steps', type=int, help='The number of episode steps in the testing.')
    parser_dynReach.add_argument('test_terminal', type=float, help='The terminal measure for testing.')
    
    #variable target
    parser_varTarg = subparsers.add_parser('varTarg', help='The variable target task.')
    parser_varTarg.add_argument('measure', type=int, help='The number of states to observe.')
    parser_varTarg.add_argument('train_steps', type=int, help='The number of steps in a training episode.')
    parser_varTarg.add_argument('train_samples', type=int, help='The total number of samples to train on.')
    parser_varTarg.add_argument('train_bound', type=float, help='The coefficient for the reward measurement.')
    parser_varTarg.add_argument('train_terminal', type=float, help='The terminal measure for the training.')
    parser_varTarg.add_argument('test_steps', type=int, help='The number of episode steps in the testing.')
    parser_varTarg.add_argument('test_terminal', type=float, help='The terminal measure for testing.')
    parser_varTarg.add_argument('test_repeat', type=int, help='The number of times to repeat the test.')
    
    #variable trajectory
    parser_varTraj = subparsers.add_parser('varTraj', help='The variable trajectory task.')
    parser_varTraj.add_argument('measure', type=int, help='The number of states to observe.')
    parser_varTraj.add_argument('train_steps', type=int, help='The number of steps in a training episode.')
    parser_varTraj.add_argument('train_samples', type=int, help='The total number of samples to train on.')
    parser_varTraj.add_argument('train_bound', type=float, help='The coefficient for the reward measurement.')
    parser_varTraj.add_argument('test_steps', type=int, help='The number of episode steps in the testing.')


    """
    parser.add_argument('--actor_hidden', metavar='N', type=int, nargs='+', default=[100,100], help='The hidden network structure for the actor')
    parser.add_argument('--actor_activation', metavar='act', type=str, choices=['relu','tanh'], default='relu', help='The activation function for the actor')
    parser.add_argument('--critic_hidden', metavar='N', type=int, nargs='+', default=[100,100], help='The hidden network structure for the critic')
    parser.add_argument('--critic_activation', metavar='act', type=str, choices=['relu','tanh'], default='relu', help='The activation function for the critic')
    """

    """
    #run the training given the arguments
    manip = args.manipulator
    state = args.state
    task = args.task
    if task == 'varTraj': #varTraj has no terminal states
        training_info = {'samples': args.samples, 'steps': args.steps, 'bound': 30/100}
        testing_info = {'steps': 500}
    else:
        training_info = {'samples':args.samples, 'steps':args.steps, 'bound':30/100, 'terminal':2/100}
        testing_info = {'steps': args.steps, 'terminal': 5/100}
    network_info = {'actor_hidden':args.actor_hidden,'actor_act':args.actor_activation,'critic_hidden':args.critic_hidden,'critic_act':args.critic_activation}
    """
    #functions to create the configs from the arguments
    #delay the imports a bit
    dynReach_config = lambda args: ConfigHandler.setup(args.manipulator,
                                                       'dynReach',
                                                       args.measure,
                                                       args.train_steps,
                                                       args.train_samples,
                                                       args.train_bound,
                                                       args.test_steps,
                                                       'relu',
                                                       [100,100],
                                                       'relu',
                                                       [100,100],
                                                       train_terminal = args.train_terminal,
                                                       test_terminal = args.test_terminal)
    
    varTarg_config =  lambda args: ConfigHandler.setup(args.manipulator,
                                                       'varTarg',
                                                       args.measure,
                                                       args.train_steps,
                                                       args.train_samples,
                                                       args.train_bound,
                                                       args.test_steps,
                                                       'relu',
                                                       [100,100],
                                                       'relu',
                                                       [100,100],
                                                       train_terminal = args.train_terminal,
                                                       test_terminal = args.test_terminal,
                                                       test_repeat = args.test_repeat)
    
    varTraj_config =  lambda args: ConfigHandler.setup(args.manipulator,
                                                       'varTraj',
                                                       args.measure,
                                                       args.train_steps,
                                                       args.train_samples,
                                                       args.train_bound,
                                                       args.test_steps,
                                                       'relu',
                                                       [100,100],
                                                       'relu',
                                                       [100,100])
    
    parser_dynReach.set_defaults(func=dynReach_config)
    parser_varTarg.set_defaults(func=varTarg_config)
    parser_varTraj.set_defaults(func=varTraj_config)
    
    args = parser.parse_args()
    config = args.func(args)    
    
    #define the config    
    #config = ConfigHandler.setup(act_type, task_type,state_measure,train_steps,train_samples,train_bound,test_steps, actor_act,actor_hidden,critic_act,critic_hidden, **kwargs)
    
    #start the controller
    from Controller import Controller #delay the import a bit
    controller = Controller(config)
    #run the training and testing and save the results
    controller.run_training(save = True).run_testing(save = True)

