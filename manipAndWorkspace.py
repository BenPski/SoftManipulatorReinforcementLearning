"""
simply a module to load the manipulators and workspaces from
probably could organize this better, but oh well
"""

import Manipulators as M
import Workspace as W
import os
import pickle

manipCable = M.CableManipulator(10,0.01,0.1)

manipTCA = M.TCAManipulator(5,0.1,3)

manips = {'cable':manipCable, 'tca':manipTCA}

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
staticWorkspaceCable = getStaticWorkspace(manipCable,cable_static_workspace)
staticWorkspaceTCA = getStaticWorkspace(manipTCA,tca_static_workspace, STEPS = 200)
dynamicWorkspaceCable = getDynamicWorkspace(manipCable,staticWorkspaceCable,cable_dynamic_workspace)
dynamicWorkspaceTCA = getDynamicWorkspace(manipTCA,staticWorkspaceTCA,tca_dynamic_workspace, SAMPLES = 50, STEPS = 100)

workspaces = {'cable_static':staticWorkspaceCable,
              'cable_dynamic':dynamicWorkspaceCable,
              'tca_static':staticWorkspaceTCA,
              'tca_dynamic':dynamicWorkspaceTCA}
