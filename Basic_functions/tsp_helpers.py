import numpy as np
from scipy.io import loadmat    #So that we can load in MATLAB files.
from matplotlib.path import Path    #So that we can reproduce MATLAB's inpolygon function.
import matplotlib.pyplot as plt #So that we can plot any borders or points that we want to visualize.
from itertools import combinations  #So that we can generate all combinations of stops for the traveling salesman problem.
import networkx as nx    #So that we can use graph algorithms to solve the traveling salesman problem.
from scipy.sparse import lil_matrix, vstack    #So that we can create sparse matrices for the constraint setting in the traveling salesman problem.
from scipy.optimize import milp #This is our mixed integer linear program solver
from scipy.optimize import LinearConstraint #This is our constraint object which said solver needs as an argument.
from scipy.optimize import Bounds #This is our bounds object which the milp solver needs as an argument.

def add_subtour_constraints(A, b, rowIdx, subTours, idxs):

    num_subtours = len(subTours)
    #Allocate the data for the new constraint on the current subtours.
    b = np.concatenate([b, np.zeros([num_subtours], dtype=np.int64)])  
    #This is scipy.sparse.vstack by the way. Not numpy.vstack. We need to set the format as "lil" 
    #because default output of coo_matrix does not support item assignment.
    A = vstack(
        [A, lil_matrix((num_subtours, idxs.shape[0]), dtype=np.int64)],
        format="lil"
    )   

    #This friend i will go through each of the current subtours we need to deal with.
    for i in range(num_subtours):

        #List all the nodes in the current subtour
        subTourIdx = list(subTours[i]) 
        #List all possible pairs of stops within the current subtour
        variations = np.array(list(combinations(range(len(subTourIdx)), 2)))   

        #This j iterates through the variations of the subtour edges.
        for j in range(len(variations)):

            #The numpy row version of the edge between the two nodes at this subtour edge variation.
            edge = np.array([subTourIdx[variations[j][0]], subTourIdx[variations[j][1]]])
            #Reorder the edge so that the smaller node comes first
            edge = np.sort(edge)

            #Where does this edge show up in the whole idxs array?
            edge_idx = np.where((idxs[:,0] == edge[0]) & (idxs[:,1] == edge[1]))[0]

            #For the subgroup associated with the given row of A, the edge_edx is among the edges we want to limit to avoid this subgroup from forming.
            A[rowIdx, edge_idx] = 1  
        #Set the corresponding bound for this subtour in b
        b[rowIdx] = len(subTourIdx) - 1

        #Move to the next row for the next subtour
        rowIdx += 1

    return A, b, rowIdx

def solve_with_subtour_constraints(Aeq, Beq, A, b, distances, idxs, bounds_obj, pr='y'):
    #Make sure the print parameter is y or n
    if pr not in ['y', 'yg', 'n']:
        raise ValueError("print parameter must be 'y' or 'n'")

    AeqA = vstack(
        [Aeq, A],
        format = "lil"
    )

    #No lower bounds for the inequality constraints, so we set them to zero
    bl = np.concatenate(
        [Beq, np.zeros((A.shape[0]))]
    )

    #The inequality constraints act as upper bounds.
    bu = np.concatenate(
        [Beq, b]
    )

    #Make sure our sizes are working together okay
    assert bl.shape[0] == bu.shape[0]
    assert AeqA.shape[0] == bl.shape[0]

    #Create a linear constraint object for the combined equality and inequality constraints
    lc = LinearConstraint(AeqA, bl, bu, keep_feasible=True)

    #And cross our fingers and hand it off to the MILP solver
    optimalLessSubtours = milp(c=distances, integrality=np.ones(idxs.shape[0]), bounds=bounds_obj, constraints=lc)

    #Let's check that Ax is within the bounds specified by the linear constraint
    if pr in ['y', 'yg']:
        Ax = Aeq @ optimalLessSubtours.x
        print("Ax:", Ax)
        print("Beq:", Beq)
        print("Ax within bounds:", np.all(np.round(Ax, 1) == Beq))

        #Let's also check the inequality constraints
        Ax_ineq = (np.round((AeqA @ optimalLessSubtours.x),1)).astype(int)  # Convert to a dense array for easier manipulation
        print("Ax_ineq:", Ax_ineq)
        print("bu:", bu)
        print("Ax_ineq within bounds:", np.all(Ax_ineq >= bl) and np.all(Ax_ineq <= bu))

    tempGSol = nx.Graph()
    tempGSol.add_edges_from(idxs[np.round(optimalLessSubtours.x, 1).astype(bool)])

    if max(dict(tempGSol.degree()).values()) > 2:
        worked = False

    else:
        worked = True

    #Took out the plotting of the subtours for simplicity

    subTours = list(nx.connected_components(tempGSol))

    if pr in ['y', 'yg']:
        for i, subtour in enumerate(subTours):
            print(f"Subtour {i+1}: {subtour}")

    return tempGSol, subTours, worked

def generateStops(num_stops, seed=None, border=None):
    if border is not None:
        if not isinstance(border, Path):
            if type(border) is not tuple:
                raise TypeError("border must be either a matplotlib.path.Path or a tuple")
            else:
                border = Path(border)

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    stops = []
    while len(stops) < num_stops: #While we haven't generated enough stops, keep generating new stops.
        candidate = np.array([  #Generate a candidate stop by randomly sampling from a uniform distribution over the bounding box of the US border.
            rng.random() * 1.5, #The x-coordinate is sampled from a uniform distribution over the range [0, 1.5], which is the bounding box of the US border.
            rng.random()    #The y-coordinate is sampled from a uniform distribution over the range [0, 1]
        ])

        if not border or border.contains_point(candidate): #Keep the candidate stop if it is inside the US border, otherwise discard it and generate a new candidate.
            stops.append(candidate)

    return np.array(stops)

#The get_border function assumes that the border comes as a MATLAB .mat file containing the coordinates. Default will be the "usborder.mat" file which you already have.

def get_border(fileName=None):
    if fileName is None:
        fileName = "Basic_functions/usborder.mat"
    border_data = loadmat(fileName)   #Loads in the MATLAB US Border file which they use to demonstrate solving this problem.

    x = border_data["x"].ravel()    #Convert the objects into 1D arrays for use in the optimization problem.
    y = border_data["y"].ravel()

    print(border_data.keys())    #See what the MATLAB file contains. The keys are the variable names, and the values are the data.
    return [Path(np.column_stack((x, y))), x, y]    #Return a Path object representing the US border.

def printGraph(stops, GSol, figsize=(10, 6), x=None, y=None):
    stops_lon = stops[:, 0]   #Extract the longitude (x-coordinate) of each stop from the first column of the stops array. We'll need these coordinates to plot the stops on a map of the US border.
    stops_lat = stops[:, 1]   #Extract the latitude (y-coordinate) of each stop from the second column of the stops array.
    plt.figure(figsize=figsize) #Create a new figure with a specified size
    if x is not None and y is not None:
        plt.plot(x, y, 'k-')  #Plot the US border using the x and y coordinates which we got from the us_border object which we created from the MATLAB file.
    plt.scatter(stops_lon, stops_lat, s=12) #Plot the generated stops as a scatter plot, with a specified marker size.
    plt.axis("equal")   #Set the aspect ratio of the plot to be equal, so that the x and y axes are scaled equally. So that the US border is recognizable and not distorted.

    for edge in GSol.edges():
        plt.plot([stops_lon[edge[0]], stops_lon[edge[1]]], [stops_lat[edge[0]], stops_lat[edge[1]]], 'b-')

    plt.show()

def Solve_With_Subtour_Open(Aeq, Beq, A=None, b=None, distances=None, idxs=None, bounds_obj=None, startNode=None, endNode=None,pr='y'):
    #Make sure the print parameter is y or n
    if pr not in ['y', 'yg', 'n']:
        raise ValueError("print parameter must be 'y' or 'n'")

    #Make sure that the A and b objects are correct size, if they are given.
    #Only then can you safely combine them with the equality constraints.
    if A is not None:
        assert A.shape[1] == Aeq.shape[1], "A must have the same number of columns as Aeq"
        AeqA = vstack(
            [Aeq, A],
            format = "lil"
        )
    else:
        AeqA = Aeq

    #For this version, we actually need to change the lower bound for Beq to be a vector of ones; beginning and ending nodes willl have just one edge.
    BeqOnes = np.ones_like(Beq)

    if b is not None:
        assert b.shape[0] == A.shape[0], "b must have the same number of rows as A"
        #No lower bounds for the inequality constraints, so we set them to zero
        bl = np.concatenate(
            [BeqOnes, np.zeros((A.shape[0]))]
        )
        #The upper bounds are all 2's, plus the constraints of the observed subtours.
        bu = np.concatenate(
            [Beq, b]
        )
    else:
        bl = BeqOnes
        bu = Beq

    #Everything above -- minus one tweak to bl, was already present appending the inequality constraints as subtour constraints.
    #However, one tweak we made afterwards was that in this function, the subtour constraints may or may not be present. hence the "if A is not None", ditto with "if b is not None"
    #Everything below is related to making the TSP problem open.
    #First, that the total number of edges must now be one less than the number of nodes, since the tour is open.
    #Recover the number of stops (nodes)
    num_stops = Aeq.shape[0]
    #Append a row of ones to the combined equality and inequality constraint matrix to account for the open tour requirement.
    AeqA = vstack(
        [AeqA, np.ones((1, AeqA.shape[1]))],
        format="lil"
    )
    bl = np.concatenate(
        [bl, np.array([num_stops - 1])]
    )
    bu = np.concatenate(
        [bu, np.array([num_stops - 1])]
    )
    if startNode is not None:
        #Make sure that the start node is a valid number
        if not (0 <= startNode < num_stops):
            raise ValueError("startNode must be between 0 and num_stops-1")
        #Set the upper bound for the start node to 1, ensuring it is at one end of the tour
        bu[startNode] = 1

    if endNode is not None:
        #Make sure that the end node is a valid number
        if not (0 <= endNode < num_stops):
            raise ValueError("endNode must be between 0 and num_stops-1")
        #Set the upper bound for the end node to 1, ensuring it is at the other end of the tour
        bu[endNode] = 1

    #Make sure our sizes are working together okay
    assert bl.shape[0] == bu.shape[0]
    assert AeqA.shape[0] == bl.shape[0]

    #Create a linear constraint object for the combined equality and inequality constraints
    lc = LinearConstraint(AeqA, bl, bu, keep_feasible=True)

    #And cross our fingers and hand it off to the MILP solver
    optimalLessSubtours = milp(c=distances, integrality=np.ones(idxs.shape[0]), bounds=bounds_obj, constraints=lc)

    tempGSol = nx.Graph()
    tempGSol.add_edges_from(idxs[np.round(optimalLessSubtours.x, 1).astype(bool)])

    if max(dict(tempGSol.degree()).values()) > 2 and min(dict(tempGSol.degree()).values()) < 1:
        worked = False

    else:
        worked = True

    #Let's check that Ax is within the bounds specified by the linear constraint
    if pr in ['y', 'yg']:
        Ax = Aeq @ optimalLessSubtours.x
        print("Ax:", Ax)

    if max(dict(tempGSol.degree()).values()) > 2:
        worked = False

    else:
        worked = True

    #Took out the plotting of the subtours for simplicity

    subTours = list(nx.connected_components(tempGSol))

    if pr in ['y', 'yg']:
        for i, subtour in enumerate(subTours):
            print(f"Subtour {i+1}: {subtour}")

    return tempGSol, subTours, worked

def full_tsp(stops, open=False, endpoints=None):
    #Make sure that the endpoints the user gave, if they gave any, are valid.
    if endpoints is not None:
        assert len(endpoints) <= 2, "Endpoints should be a list or tuple of at most two elements."
        assert all(0 <= e < stops.shape[0] for e in endpoints), "Endpoints should be valid stop indices."

    if endpoints is not None:
        startNode = endpoints[0]
        endNode = endpoints[1] if len(endpoints) == 2 else None
    else:
        startNode, endNode = None, None

    n_stops = stops.shape[0]    #Get the number of stops from the shape of the stops array.

    #Now we get to generate a list of all combinations of stops that we want to consider for the traveling salesman problem. We will use these combinations to generate a distance matrix, which we will then use to solve the traveling salesman problem.

    #Combinations lists all possible pairs of stops from the list of stops and number we want of stops per combination (2; start and finish). The output is a list of tuples, where each tuple contains the indices of the two stops in the combination. We convert this list of tuples into a NumPy array for easier manipulation and analysis.
    idxs = np.array(list(combinations(range(len(stops[:,0])), 2)))    
    #Calculate the Euclidean distance between each pair of stops. The differences between the coordinates of the stops are computed, and then the norm (distance) is calculated along the specified axis (axis=1 for row-wise operation).
    distances = np.linalg.norm(stops[idxs[:,0], :] - stops[idxs[:,1], :], axis=1)   

    #Now we get to create a sparse matrix to represent which cities are included in each edge, which begins for us the constraint setting.

    Aeq = lil_matrix((n_stops, idxs.shape[0]), dtype=np.int8)   #Create a sparse matrix where each row represents a city and each column represents an edge, with entries indicating whether a city is included in an edge.

    #Now we need to populate the matrix using a for loop
    for i in range(n_stops):    #For each city,
        which_idxs = np.where((idxs[:,0] == i) | (idxs[:,1] == i))[0]    #Find the indices of the edges that include the current city.
        Aeq[i, which_idxs] = 1    #Set the entries corresponding to the edges that include the current city to 1.

    Beq = 2*np.ones(n_stops, dtype=np.int64) #This creates a vector of 2's to help us set the constraint of edges connected per city.

    #Make sure we create the lower and upper bounds object correctly to be used in the optimize problem.
    lb = np.zeros(idxs.shape[0], dtype=np.int64)   #Lower bound for the decision variables, typically set to 0.
    ub = np.ones(idxs.shape[0], dtype=np.int64)    #Upper bound for the decision variables, typically set to 1.

    #This linear constraint should say that b <= Aeq * x <= b, which I believe is the correct way to represent equality constraints in the form of a linear constraint.
    bounds_binary = np.array([lb, ub]).T
    bounds_binary

    lc = LinearConstraint(Aeq, Beq, Beq, keep_feasible=True)

    bounds_obj = Bounds(lb, ub)

    #Do we mean to solve open, or closed?
    if open:
        [tempGSol, subTours, worked] = Solve_With_Subtour_Open(Aeq, Beq, A=None, b=None, distances=distances, idxs=idxs, bounds_obj=bounds_obj, startNode=None, endNode=None, pr='n')

    else:
        #Solve it as is, which creates subtours.
        optimal_subtours = milp(c=distances, integrality=np.ones(idxs.shape[0]), bounds=bounds_obj, constraints=lc)
        tempGSol = nx.Graph()
        tempGSol.add_edges_from(idxs[np.round(optimal_subtours.x, 1).astype(bool)])

    A = lil_matrix((0,idxs.shape[0]), dtype=np.int8) 
    #This array will hold the bounds for how many of the edges from matrix A we allow, effectively restricting loops to form among those subtour groups.
    b = np.array([], dtype=np.int8) 

    #Our friend here will count through the rows of A and b as we make them.
    rowIdx=0

    num_subtours = nx.number_connected_components(tempGSol)
    subTours = list(nx.connected_components(tempGSol))

    while len(subTours)>1:    
        [A, b, rowIdx] = add_subtour_constraints(A, b, rowIdx, subTours, idxs)
        if not open:
            [tempGSol, subTours, worked] = solve_with_subtour_constraints(Aeq, Beq, A, b, distances, idxs, bounds_obj)
        else:
            [tempGSol, subTours, worked] = Solve_With_Subtour_Open(Aeq, Beq, A, b, distances, idxs, bounds_obj, startNode=startNode, endNode=endNode)
    return tempGSol, worked

def stops_edge_idxs(stops):

    n_stops = stops.shape[0]

    #Combinations lists all possible pairs of stops from the list of stops and number we want of stops per combination (2; start and finish). The output is a list of tuples, where each tuple contains the indices of the two stops in the combination. We convert this list of tuples into a NumPy array for easier manipulation and analysis.
    idxs = np.array(list(combinations(range(len(stops[:,0])), 2)))    
    #Calculate the Euclidean distance between each pair of stops. The differences between the coordinates of the stops are computed, and then the norm (distance) is calculated along the specified axis (axis=1 for row-wise operation).
    distances = np.linalg.norm(stops[idxs[:,0], :] - stops[idxs[:,1], :], axis=1) 
    Aeq = lil_matrix((n_stops, idxs.shape[0]), dtype=np.int8)   #Create a sparse matrix where each row represents a city and each column represents an edge, with entries indicating whether a city is included in an edge.  
    #Now we need to populate the matrix using a for loop
    for i in range(n_stops):    #For each city,
        which_idxs = np.where((idxs[:,0] == i) | (idxs[:,1] == i))[0]    #Find the indices of the edges that include the current city.
        Aeq[i, which_idxs] = 1    #Set the entries corresponding to the edges that include the current city to 1.

    return idxs, distances, Aeq

def filter_edges(idxs, distances, Aeq, filter_size):
    can_keep = []


    for i in range(Aeq.shape[0]):
        indices = (Aeq[i,:]==1).indices

        our_distances = distances[indices]

        #These are the indices (within the array of just edges including our case node) that we want to keep based on the smallest distances
        keep_distances = np.argsort(our_distances)[:filter_size]

        #Now we set them to "True" in the can_keep object.
        keep_indices = indices[keep_distances]

        can_keep.extend(keep_indices[~np.isin(keep_indices, can_keep)])

    return idxs[can_keep],  distances[can_keep], Aeq[:, can_keep]