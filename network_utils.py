from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
import rl.random


def generateAgent(env, actor_hidden, actor_act, critic_hidden, critic_act):
    nb_actions = env.action_space.shape[0]
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    for h in actor_hidden:
        actor.add(Dense(h))
        actor.add(Activation(actor_act))
    actor.add(Dense(nb_actions))
    actor.add(Activation('sigmoid')) # get output to go from 0 to 1

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate(axis=-1)([action_input, flattened_observation])
    for h in critic_hidden:
        x = Dense(h)(x)
        x = Activation(critic_act)(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(input=[action_input, observation_input], output=x)

    memory = SequentialMemory(limit=500000, window_length=1)
    random_process = rl.random.OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                      random_process=random_process, gamma=0.99, target_model_update=1e-3)
    agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

    return agent
