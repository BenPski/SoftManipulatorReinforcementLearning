# SoftManipulatorReinforcementLearning

This provides the code used for the reinforcement learning testing and training with soft manipulators. It mainly depends on Keras-rl and [SoftManipulatorDynamics](https://github.com/BenPski/SoftManipulatorDynamics). The other dependencies include tensorflow (at least that is what I use as the backend), numpy, and matplotlib. It seems that it is the easiest to set this up as an anaconda environment with python3.5 (I don't quite remember why 3.5 specifically, I could be wrong as well). A bit of the dependencies have to be adjusted, but it should be easy enough to get that working.

Due to the dependence on the manipuator dynamics it needs to connect with Matlab, to do this the [Matlab python api](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html). If you use anaconda to manage your python environments then this can be a bit tricky to setup due to needing to know where to install the libraries.

# Getting Matlab Python API Working

To get the Matlab api working with anaconda, first you need to determine which python version is compatible with the api, this can be 2.7, 3.5, or 3.6 (for example 2018a works with 3.6, 2016b works with 3.5). So, if you already have a python environment setup with the right python you can move on, if you don't then one will have to be created. This can be done like:
```bash
conda create -n py35 python=3.5
```
to create an environment with python 3.5.

Then, with the environment setup we can then we need to determine the location of Matlab so that we can find the correct files to install. To do this open matlab and run "matlabroot" and whatever it spits out is the root directory of matlab. 

With that we need to locate the python api setup.py file and run it for the proper environment.

Start by entering your environment, this makes sure the right python version is running. (this can vary between operating systems)
```bash
conda activate PY_ENV
```
where PY_ENV is the environment you want (e.g., it would be py35 if following from the prior step)

Now change to the proper matlab directory (some syntax changes between operating systems / vs \)
```bash
cd MATLABROOT/extern/engines/python
```
where MATLABROOT is the matlab location looked up before.

Now we can install the api into the desired environment, for this the directory structure can be quite varied, so you'll have to figure out the relevant paths.
```bash
python setup.py build --build-base="SOMEWHERE ELSE" install --prefix="ANACONDA/envs/PY_ENV"
```
Now, if you don't have root/admin access specifying a build-dir is likely necessary, it can be anywhere (like Downloads) and then just deleted later. The prefix then states where the library should be saved to. In this case it needs to be stored in the relevant anaconda environment. The PY_ENV should be the same as used before and ANACONDA is wherever your anaconda directory is, it could be named "anaconda" or "anaconda3" (for example on linux it would be "~/anaconda3/").

The api should be installed to test that is worked try to open python and import matlab.engine, if there is no error then it should work.

# Getting the Manipulators Interfaced to the Libraries
To get the code in Manipulators.py properly working the location of the SoftManipulatorDynamics needs to be known, currently the default configuration will look in the current directory. You can edit the manip_config.ini to set the location of the manipulators directory, also the program will simply fail when creating a manipulator object until the configuration is properly specified.

# Usage
The usage is pretty straightforward for the running of the experiments. It is best to operate it via RunManyExperiments which needs a config specifying what to run. An example config would look like:
```ini
[Exp Cable 1 DynReach]
manip = cable
task = dynReach
measure = 1
train_steps = 100
train_samples = 20000
train_bound = 0.3
train_terminal = 0.05
test_steps = 100
test_terminal = 0.05
replicates = 5

[Exp TCA 1 DynReach]
manip = tca
task = dynReach
measure = 1
train_steps = 500
train_samples = 100000
train_bound = 0.3
train_terminal = 0.05
test_steps = 500
test_terminal = 0.05
replicates = 5

[Exp Cable 1 VarTarg]
manip = cable
task = varTarg
measure = 1
train_steps = 100
train_samples = 200000
train_bound = 0.3
train_terminal = 0.05
test_steps = 100
test_terminal = 0.05
test_repeat = 100
replicates = 5

[Exp TCA 1 VarTarg]
manip = tca
task = varTarg
measure = 1
train_steps = 500
train_samples = 1000000
train_bound = 0.3
train_terminal = 0.05
test_steps = 500
test_terminal = 0.05
test_repeat = 100
replicates = 5

[Exp Cable 1 VarTraj]
manip = cable
task = varTraj
measure = 1
train_steps = 100
train_samples = 2000000
train_bound = 0.3
test_steps = 500
replicates = 5

[Exp TCA 1 VarTraj]
manip = tca
task = varTarg
measure = 1
train_steps = 500
train_samples = 2000000
train_bound = 0.3
test_steps = 500
replicates = 5
```

Then running the experiments looks like:
```bash
python RunManyExperiments.py learning_experiments.ini
```

For the performances, the automation still is being implemented, but can be easily grabbed by initializing a ConfigHandler from saved configuration file and then using load_perf().

```python
from ConfigHandler import ConfigHandler

config = ConfigHandler.fromFile('./test/___.ini')
perf = config.load_perf()
```
