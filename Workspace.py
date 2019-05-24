import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


"""

for the workspace, I am using a convex hull to define the workspace

the important things to include for the workspace are whether or not a value is inside the workspace and being able to sample from the workspace

"""

class Workspace(object):
    """
    a workspace is a convex hull
    in this case pass in the sample points and the hull will be formed
    """

    def __init__(self, points):
        self.points = points
        self.hull = ConvexHull(points)
        #want to store the  min and max points for the random sampling
        self.min = np.min(points,0)
        self.max = np.max(points,0)

        #some useful definitions
        self.vertices = self.points[self.hull.vertices]

        self.max_x,self.max_y,self.max_z = np.max(self.vertices,0)
        self.min_x,self.min_y,self.min_z = np.min(self.vertices,0)

        self.max_r = np.max(np.linalg.norm(self.vertices[:,:2],axis=1)) #the maximum radius


    def inside(self,x):
        #check if a point is inside the hull
        #could probably be more efficient about this
        for eq in self.hull.equations:
            if np.dot(x,eq[:-1])+eq[-1] > 0:
                return False
        return True

    def sample(self):
        #generate a random sample from inside the workspace
        #hard to define inside, so sample the maximum cube until the point is inside the hull
        p = np.random.uniform(self.min,self.max)
        while not self.inside(p):
            p = np.random.uniform(self.min,self.max)
        return p

    def plot(self,ax=None,color='b'):
        #generate a plot of the hull
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        for s in self.hull.simplices:
            s = np.append(s,s[0])
            ax.plot(self.points[s,0], self.points[s,1], self.points[s,2], color=color, alpha=0.3)
        #plt.show()
        return ax
        
    def surface(self,ax=None,color='b',alpha=0.2):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        p = Poly3DCollection([self.hull.points[face] for face in self.hull.simplices], alpha=alpha)
        p.set_facecolor(color)
        ax.add_collection3d(p)
        
        return ax
        


def makeStaticManipulatorWorkspace(manip, SAMPLES = 100, STEPS = 500):
    """
    approximate the static workspace of a manipulator

    SAMPLES: points to generate
    STEPS: steps to run dynamics to achieve steady state
    """

    #random actions
    #acts = np.random.uniform([0,0,0],manip.max_q*np.array([1,1,1]),(SAMPLES,3))

    #grid of actions
    # the actions are automatically scaled in the manipulator object
    vals = np.linspace(0,1,round(SAMPLES**(1/3)))
    acts = []
    for act1 in vals:
        for act2 in vals:
            for act3 in vals:
                acts.append(np.array([act1,act2,act3]))
    acts = np.array(acts)

    points = np.zeros(acts.shape)

    for s in range(acts.shape[0]):
        manip.initState()
        a = acts[s,:]
        #for _ in range(5): #first couple steps are small to see if it improves integration
        #    manip.step(a/10)
        for _ in range(STEPS):
            manip.step(a)
            manip.render(hold=True)
    
        state = manip.getState()
        g = state['g']
        #pos = g[9:12,-1]
        pos = g[12:15,-1]
        points[s,:] = pos
        

    return Workspace(points)

def makeStaticManipulatorWorkspaceFromStatics(manip, SAMPLES = 100):
    """
    These names are getting really ridiculous
    
    get the static workspace by running the actual statics

    SAMPLES: points to generate
    """

    #random actions
    #acts = np.random.uniform([0,0,0],manip.max_q*np.array([1,1,1]),(SAMPLES,3))

    #grid of actions
    # the actions are automatically scaled in the manipulator object
    vals = np.linspace(0,1,round(SAMPLES**(1/3)))
    acts = []
    for act1 in vals:
        for act2 in vals:
            for act3 in vals:
                acts.append(np.array([act1,act2,act3]))
    acts = np.array(acts)

    points = np.zeros(acts.shape)
    manip.initState()
    for s in range(acts.shape[0]):
        a = acts[s,:]
        manip.step(a)
        manip.render(hold=True)
    
        state = manip.getState()
        g = state['g']
        pos = g[9:12,-1]
        points[s,:] = pos
        

    return Workspace(points)

def makeDynamicManipulatorWorkspace(manip, SAMPLES = 500, STEPS = 5, IDLE_CHANCE = 0.5, RESETS = 1, render=False):
    """
    approximate the dynamic workspace of a manipulator

    do this by randomly exploring the workspace (not sure of another way)
    in this case collect every sample point generated and want to allow the actuation to run for a few steps in a given direction
    also have the chance to idle

    want to try to get a good sampling of the workspace as possible
    """
    points = []
    for _ in range(RESETS):
        manip.initState()
        for _ in range(SAMPLES):
            idle = np.random.rand()
            if idle < IDLE_CHANCE:
                a = np.array([0,0,0])
            else:
                a = np.random.uniform([0,0,0],np.array([1,1,1]))
            for _ in range(STEPS):
                manip.step(a)
                if render:
                    manip.render(hold=True)
                state = manip.getState()
                g = state['g']
                #pos = g[-9:12,-1]
                pos = g[12:15,-1]
                points.append(pos)

    return Workspace(np.array(points))

def dynamicWorkspace(manip, staticWorkspace, SAMPLES = 500, STEPS = 5, IDLE_CHANCE = 0.5, render = False, MAX_ITERS = 20):
    """
    the dynamic workspace should enclose the static workspace
    so, run the dynamics until the vertices of the static workspace are all enclosed in the dynamic workspace
    """
    dynWorkspace = Workspace(staticWorkspace.vertices) #possible that this satisfies the condition since inside is subject to floating point truncation, just need some initialization
    iters = 0
    while (not all([dynWorkspace.inside(v) for v in staticWorkspace.vertices])) and (iters < MAX_ITERS):
        #print([dynWorkspace.inside(v) for v in staticWorkspace.vertices])
        manip.initState()
        points = []
        for _ in range(SAMPLES):
            idle = np.random.rand()
            if idle < IDLE_CHANCE:
                a = np.array([0,0,0])
            else:
                a = np.random.uniform([0,0,0],[1,1,1])
            for _ in range(STEPS):
                manip.step(a)
                if render:
                    manip.render(hold=True)
                state = manip.getState()
                g = state['g']
                pos = g[9:12,-1]
                points.append(pos)
        all_points = points + dynWorkspace.vertices.tolist()
        dynWorkspace = Workspace(np.array(all_points))
        iters += 1
    return dynWorkspace



def buldgeWorkspace(workspace, r):
    """
    if it is desired to have a slightly larger workspace in all directions

    to do it sample every vertex in the original workspace
        at every sample add on spherical points around the vertex
        create a new convex hull based on the points
    """

    points = []
    for vertex in workspace.vertices:
        for theta in np.linspace(0,np.pi,10):
            for phi in np.linspace(0,2*np.pi,10):
                points.append(vertex+np.array([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta)]))
    return Workspace(np.array(points))
