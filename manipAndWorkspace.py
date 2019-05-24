"""
simply a module to load the manipulators and workspaces from
probably could organize this better, but oh well
"""

import Manipulators as M
import Workspace as W
import os
import pickle

#make them lazyily loaded to avoid starting extra matlab instances that don't get closed
#manipCable = M.CableManipulator(10,5e-3,0.5)
manipCable = lambda : M.CableManipulator(6,0.01,-0.5)
manipCableStatic = lambda : M.CableManipulator(6,0.1,-0.5) #long step to get nearly static
#manipCableStatic = lambda : M.CableManipulatorStatic(6,-0.5)

#manipTCA = M.TCAManipulator(5,0.1,3)



#the workspace file locations
def getStaticWorkspace(manip,filename, **kwargs):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        #workspace = W.makeStaticManipulatorWorkspace(manip, **kwargs)
        manip = manip()
        workspace = W.makeStaticManipulatorWorkspace(manip, **kwargs)
        #workspace = W.makeStaticManipulatorWorkspaceFromStatics(manip, **kwargs)
        with open(filename, 'wb') as f:
            pickle.dump(workspace, f)
        return workspace

def getDynamicWorkspace(manip, staticWorkspace, filename, **kwargs):
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        manip = manip()
        workspace = W.makeDynamicManipulatorWorkspace(manip, **kwargs)
        #workspace = W.dynamicWorkspace(manip, staticWorkspace, **kwargs)
        with open(filename, 'wb') as f:
            pickle.dump(workspace, f)
        return workspace

cable_static_workspace = "workspaces/cable_static_workspace.pkl"
tca_static_workspace = "workspaces/tca_static_workspace.pkl"

cable_dynamic_workspace = "workspaces/cable_dynamic_workspace.pkl"
tca_dynamic_workspace = "workspaces/tca_dynamic_workspace.pkl"

#only need to use the tip state
#staticWorkspaceCable = getStaticWorkspace(manipCableStatic,cable_static_workspace)
staticWorkspaceCable = getStaticWorkspace(manipCableStatic,cable_static_workspace, STEPS=10)
#staticWorkspaceTCA = getStaticWorkspace(manipTCA,tca_static_workspace, STEPS = 200)
dynamicWorkspaceCable = getDynamicWorkspace(manipCable,staticWorkspaceCable,cable_dynamic_workspace, IDLE_CHANCE = 0.2, STEPS = 10)
#dynamicWorkspaceTCA = getDynamicWorkspace(manipTCA,staticWorkspaceTCA,tca_dynamic_workspace, SAMPLES = 50, STEPS = 100)



#the more convenient things to be exporting and using
"""
workspaces = {'cable':{'static':staticWorkspaceCable, 
                       'dynamic':dynamicWorkspaceCable}, 
              'tca':{'static':staticWorkspaceTCA, 
                     'dynamic':dynamicWorkspaceTCA}
              }

manips = {'cable':manipCable, 'tca':manipTCA}
"""
workspaces = {'cable':{'static':staticWorkspaceCable, 
                       'dynamic':dynamicWorkspaceCable}
              }
manips = {'cable':manipCable}