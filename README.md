# SoftManipulatorReinforcementLearning

This provides the code used for the reinforcement learning testing and training with soft manipulators. It depends on Keras-RL and SoftManipulatorDynamics. 

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
python build --build_dir="SOMEWHERE ELSE" install --prefix="ANACONDA/envs/PY_ENV"
```
Now, if you don't have root/admin access specifying a build-dir is likely necessary, it can be anywhere (like Downloads) and then just deleted later. The prefix then states where the library should be saved to. In this case it needs to be stored in the relevant anaconda environment. The PY_ENV should be the same as used before and ANACONDA is wherever your anaconda directory is, it could be named "anaconda" or "anaconda3" (for example on linux it would be "~/anaconda3/").

The api should be installed to test that is worked try to open python and import matlab.engine, if there is no error then it should work.

# Getting the Manipulators Interfaced to the Libraries
To get the code in Manipulators.py properly working the location of the SoftManipulatorDynamics need to be known, this can be specified when contructing the Manipulators object or it can be hardcoded elsewhere.

