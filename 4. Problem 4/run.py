# Necessary packages
import os # execute terminal commands
import numpy as np
import torch # device assignment for importance model adn GPU number assignment in Hydra zen Joblib
from pathlib import Path # path manipulation for mlflow
import time # calculate approximate training time

# Modulus specific plot
from sympy import Symbol, Eq, Function, Number, Abs # defining symbols
import modulus # load classes and functions from modulus directory
from modulus.hydra import to_absolute_path, to_yaml, instantiate_arch  # configure settings and create architecture
from modulus.hydra.config import ModulusConfig # load config files
#from modulus.csv_utils.csv_rw import csv_to_dict  # load true solution for validation
from modulus.continuous.solvers.solver import Solver # continuous time solver
from modulus.continuous.domain.domain import Domain  # domain (custom classes + architecture)
from modulus.geometry.csg.csg_2d import Rectangle  # CSG geometry
from modulus.continuous.constraints.constraint import (  # constraints
    PointwiseBoundaryConstraint,  # BC
    PointwiseInteriorConstraint,  # interior/collocation points
)
from modulus.constraint import Constraint # don't know
from modulus.continuous.validator.validator import PointwiseValidator  # adding validation dataset
from modulus.continuous.inferencer.inferencer import PointwiseInferencer # infer variables
from modulus.tensorboard_utils.plotter import ValidatorPlotter, InferencerPlotter # tensorboard
from modulus.key import Key  # add keys to nodes
from modulus.node import Node # add node to geometry

from modulus.pdes import PDES # PDEs
from modulus.graph import Graph # for defining importance graph model
from modulus.architecture.fully_connected import FullyConnectedArch
from modulus.architecture.fourier_net import FourierNetArch
from modulus.architecture.siren import SirenArch
from modulus.architecture.modified_fourier_net import ModifiedFourierNetArch
from modulus.architecture.dgm import DGMArch
from modulus.architecture.layers import Activation

# Plotting packages
from itertools import product # create combinations of elements of multiple arrays
import matplotlib.pyplot as plt
from scipy.interpolate import griddata # interpolation for plt.imshow()
from mpl_toolkits.mplot3d import Axes3D # for 3D if needed
# from matplotlib import cm # colormap
from matplotlib import ticker # controls number of ticks in colorbar, helpful for very ugly unsymmetric colorbar

# Logging things
import logging # inbuilt text logging
import mlflow # sweep specific logger with csv file and multimedia
# Store print stdout in the python logger
# https://bobbyhadz.com/blog/python-assign-string-output-to-variable
from io import StringIO
import sys



# The PDE class
class HeatConductionEquation2D(PDES):
    """
    Heat Conduction 2D
    
    Parameters
    ==========
    k : float, string
        Conductivity doesn't matter in steady state problem with no heat source
    """

    name = "HeatConductionEquation2D"
    def __init__(self):
        # coordinates
        x = Symbol("x")

        # time
        y = Symbol("y")

        # parametric conductivity like term
        k = Symbol("k")
        
        # make input variables
        input_variables = {"x": x, "y": y, "k": k}

        # Temperature output
        T = Function("T")(*input_variables)

        # conductivity coefficient
        # if type(c) is str:
        #     c = Function(c)(*input_variables) # if c is function of independent variables
        # elif type(c) in [float, int]:
        #     c = Number(c)

        # set equations
        self.equations = {}
        self.equations["heat_equation"] = T.diff(x, 2) + k* T.diff(y, 2)  # diff(variable,order of derivative)


############ MAIN FUNCTION ########################
@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    
    ###################### PREPROCESSING ##########################
    # cfg.network_dir = 'outputs_1'    # Set the network directory for checkpoints outputs/2d_heat_conduction/outputs_1 so it is useless
    
    # stdout buffer
    # store anything printed by DeepXDE in a variable
    # https://bobbyhadz.com/blog/python-assign-string-output-to-variable
    buffer = StringIO()
    sys.stdout = buffer
    
    # If the cfg files is missing
    if cfg.get("error", False):
        raise RuntimeError("cfg.error is True")
    
    # Get the sweep specific directory
    sweep_specific_directory = os.getcwd()
    
    # Setup the python logger
    ## Logger for a specific sweep
    logging.basicConfig(level=logging.INFO, filemode = 'w', format='%(asctime)s :: %(message)s', filename=f'{sweep_specific_directory}/training.log',force=True) # https://stackoverflow.com/a/71625766/14598633
    logging.info(f'Logger Initialised')
    #logging.info(f'Sweep number : {sweep_number}')
    logging.info(f'Sweep directory : {sweep_specific_directory}')
    
    logging.info(f'Learning Rate : {cfg.optimizer.lr}') # this is is applied automatically
    logging.info(f'Activation : {cfg.activation}') # this one is not supported to doing it manually
    logging.info(f'Number of Layers : {cfg.layers}') # number of DGM layers          

    
    # loading activations
    if cfg.activation == 'tanh': 
        act_func = Activation.TANH
    elif cfg.activation == 'stan':
        act_func = Activation.STAN
    elif cfg.activation == 'selu':
        act_func = Activation.SILU
    elif cfg.activation == 'gelu':
        act_func = Activation.GELU
    elif cfg.activation == 'relu':
        act_func = Activation.LEAKY_RELU
    elif cfg.activation == 'sin':
        act_func = Activation.SIN
    else:
        print(sys.exit("No Activation specified"))

    # more help can be found here : https://docs.nvidia.com/deeplearning/modulus/user_guide/features/configuration.html?highlight=activation#command-line-interface
    
    # Print the config
    logging.info('########### CONFIG FILE ################## ')
    logging.info(to_yaml(cfg))
    
    
    # Logging GPU information
    #logging.info("GPU information")
    #logging.info(os.system("nvidia-smi"))
    
    #################### NVIDIA MODULUS MAIN PART #########################

    # Define parametric PDE inputs
    if cfg.custom.parameterized:
        input_keys = [Key("x"), Key("y"), Key("k")] # one more input
    else:
        input_keys=[Key("x"), Key("y")]
        
    output_keys=[Key("T")]
    
    logging.info('Detecting architecture')
    # make list of nodes to unroll graph on
    heat_eq = HeatConductionEquation2D()
    if cfg.custom.arch == "FullyConnectedArch":
        logging.info('Architecture: Fully Connected Arch')
        heat_net = FullyConnectedArch(
            input_keys=input_keys,
            output_keys=output_keys,
            layer_size=cfg.layers,
            activation_fn=act_func, # set activation function
            adaptive_activations=cfg.custom.adaptive_activations,
        )
    elif cfg.custom.arch == "FourierNetArch":
        logging.info('Architecture: Fourier Network')
        heat_net = FourierNetArch(
            input_keys=input_keys,
            output_keys=output_keys,
            adaptive_activations=cfg.custom.adaptive_activations,
            activation_fn=act_func, # set activation function
        )
    elif cfg.custom.arch == "SirenArch":
        logging.info('Architecture: SIREN Arch')
        heat_net = SirenArch(
            input_keys=input_keys,
            layer_size=cfg.layers,
            output_keys=output_keys,
            activation_fn=act_func, # set activation function
        )
    elif cfg.custom.arch == "ModifiedFourierNetArch":
        logging.info('Architecture: Modified Fourier Arch')
        heat_net = ModifiedFourierNetArch(
            input_keys=input_keys,
            output_keys=output_keys,
            layer_size=cfg.layers,
            adaptive_activations=cfg.custom.adaptive_activations,
            activation_fn=act_func, # set activation function
        )
    elif cfg.custom.arch == "DGMArch":
        logging.info('Architecture: DGM Arch')
        heat_net = DGMArch(
            input_keys=input_keys,
            output_keys=output_keys,
            layer_size=cfg.layers,
            adaptive_activations=cfg.custom.adaptive_activations,
            activation_fn=act_func, # set activation function
        )
    else:
        sys.exit(
            "Network not configured for this script. Please include the network in the script"
        )
    #print('debug network')
    #print(heat_net)
    # heat_net = instantiate_arch(
    #     input_keys=[Key("x"), Key("y")],
    #     output_keys=[Key("T")],
    #     cfg=cfg.arch.fully_connected,
    # )
    
    nodes = heat_eq.make_nodes() + [heat_net.make_node(name="heat_network", jit=cfg.jit)]
    logging.info('Nodes Initialised')
    
    # make importance model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        logging.info(f'GPU Name : {torch.cuda.get_device_name(0)}')
    
    importance_model_graph = Graph(
        nodes,
        invar=[Key("x"), Key("y"), Key("k")],
        req_names=[
            Key("T", derivatives=[Key("x")]),
            Key("T", derivatives=[Key("y")]),
        ],
    ).to(device)

    def importance_measure(invar):
        outvar = importance_model_graph(
            Constraint._set_device(invar, requires_grad=True)
        )
        importance = (
            outvar["T__x"] ** 2
            + outvar["T__y"] ** 2
        ) ** 0.5 + 10
        return importance.cpu().detach().numpy()

    
    # make geometry
    x, y, L = Symbol("x"), Symbol("y"), Symbol('k')
    lower_bound = (-0.5, 0.0)
    upper_bound = (0.5, 1.0) # y - upper bound is parametric 
    
    
    rec = Rectangle(lower_bound, upper_bound) # (x_1,y_1), (x_2,y_2)
    pr = {Symbol("k"): (0.0, 1.0)} # param_ranges for variation of parametric L
    
    logging.info('Geometry created')
          
    # make domain
    heat_domain = Domain()
    logging.info('Domain created')
          
    # Adding constraints
    # Bottom wall
    bottom_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"T": 0.0}, # outputs
        batch_size=cfg.batch_size.Wall,
        lambda_weighting={"T": 1.0 - 2 * Abs(x)},  # weight edges to be zero
        criteria=Eq(y, 0.0), # coordinates for sampling
        importance_measure=importance_measure,
        param_ranges=pr,
        
    )
    heat_domain.add_constraint(bottom_wall, "bottom_wall")
    
    logging.info('Bottom wall BC created')
          
    # Top wall
    top_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"T": 0.0}, # outputs
        batch_size=cfg.batch_size.Wall,
        lambda_weighting={"T": 1.0 - 2 * Abs(x)},  # weight edges to be zero
        criteria=Eq(y, 1.0), # coordinates for sampling
        importance_measure=importance_measure,
        param_ranges=pr,
    )
    heat_domain.add_constraint(top_wall, "top_wall")

    logging.info('Top wall BC created')
    
    # Left wall
    left_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"T": 1.0}, # outputs
        batch_size=cfg.batch_size.Wall,
        lambda_weighting={"T": 1.0 - 2 * Abs(y-0.5)},  # weight edges to be zero
        criteria=Eq(x, -0.5), # coordinates for sampling
        importance_measure=importance_measure,
        param_ranges=pr,
    )
    heat_domain.add_constraint(left_wall, "left_wall")
    
    logging.info('Left wall BC created')
          
    # Right wall
    right_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"T": 1.0}, # outputs
        batch_size=cfg.batch_size.Wall,
        lambda_weighting={"T": 1.0 - 2 * Abs(y-0.5)},  # weight edges to be zero
        criteria=Eq(x, 0.5), # coordinates for sampling
        importance_measure=importance_measure,
        param_ranges=pr,
    )
    heat_domain.add_constraint(right_wall, "right_wall")
    
    logging.info('Right wall BC created')

    # PDE constraint
    x_bound = (-0.5,0.5)
    y_bound = (0, 1)
    
    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"heat_equation" : 0},
        batch_size=cfg.batch_size.Interior,
        bounds={x: x_bound, y: y_bound}, # sampling should be from along y 0,10. see baseline PINN version for some intuition
        # lambda weights for now
        lambda_weighting={"heat_equation": rec.sdf},
        importance_measure=importance_measure,
        param_ranges=pr,
    )
    heat_domain.add_constraint(interior, "interior")
    
    logging.info('Interior points created')
    
    # Getting path (nvidia modulus only accepts absolute path to remove any ambiguity)
#     validation_path = to_absolute_path("final_data.npz")

#     # Adding VALIDATION data
#     data = np.load(validation_path) # Load data from file
#     keys = list(data.keys()) # all keys in the dictionary
#     logging.info(f'Validation dataset keys : {keys}')
#     nodes3D = data[keys[0]]
#     temperature = data[keys[1]]
#     boundary_nodal_coordinates3D = data[keys[2]]
#     boundary_solution = data[keys[3]]
#     # face3 = data[keys[4]]
#     # face4 = data[keys[5]]
#     # face5 = data[keys[6]]
#     # face6 = data[keys[7]]
    
    
#     # cutting the useless third dimension where there is nothing to predict
#     node = nodes3D[:,[0,2]] # remember there is nodes variable also, they should not clash
#     #temperature = temperature3D[:,[0,2]]
#     boundary_nodal_coordinates = boundary_nodal_coordinates3D[:,[0,2]]
#     #boundary_solution = boundary_solution3D[:,[0,2]]
    
    
#     # This is the required format for validation data
#     openfoam_invar_numpy = {}
#     openfoam_invar_numpy["x"] = node[:, 0][:, None]
#     openfoam_invar_numpy["y"] = node[:, 1][:, None]

#     openfoam_outvar_numpy = {}
#     openfoam_outvar_numpy["T"] = temperature
    
#     openfoam_validator = PointwiseValidator(
#         openfoam_invar_numpy,
#         openfoam_outvar_numpy,
#         nodes,
#         batch_size=1024,
#         plotter=ValidatorPlotter(),
#     )
#     heat_domain.add_validator(openfoam_validator)
    
    logging.info('Validation Data loaded : NA')
    
    training_dataset = np.array([]) # the training dataset
    L_ranges = np.array([0.1, 0.5, 1.0]) #np.linspace(0.0, 1.0, 11) # the 3rd input : conductivity slices
    x_ranges = np.linspace(-0.5, 0.5, 80) # x coordinates
    
    # using the old code
    for length_parameter in L_ranges: # for each length param
        # y is used to satisfy the PDE i.e the collocation points
        y_ranges = np.linspace(0.0, 1.0, 80) # y coordinates is straightforward in parametric conductivity. The 2D geometry is a cuboid. if in doubt, then plot the vtk files.

        temp_dataset = np.array(list(product(x_ranges, y_ranges, [length_parameter]))) # create an numpy from itertools.product (rectangular slices for each conductivity slice

        training_dataset = np.vstack([training_dataset, temp_dataset]) if training_dataset.size else temp_dataset # stackup the rectangular slices to make a cuboid in the end of the loop
    
    invar_numpy = {"x": training_dataset[:,0][:,None], "y": training_dataset[:,1][:,None], "k": training_dataset[:,2][:,None]} # conductivity attached to the inference dictionary
    
    # add INFERENCER data
    grid_inference = PointwiseInferencer(
        invar_numpy,
        ["T"],#, "T__x", "T__y"],
        nodes,
        batch_size=1024,
        plotter=InferencerPlotter(),
    )
    heat_domain.add_inferencer(grid_inference, "inf_data")
    
    logging.info('Inference outputs loaded')
    
        # make solver
    slv = Solver(cfg, heat_domain)

    # start solver
    start_time = time.time() # start time
    
    slv.solve()
    
    elapsed = time.time() - start_time
    
    logging.info(f'Training time(sec) : {elapsed}') # approximate training time
    
    
    ##################### PLOTTING #######################
    # If this is not here mlflow won't save the artifact, thanks to function scoping
    # plotting() # Executing plotting routine
    


    # The plotting class
    # def plotting(): # no class plz
    # what's what : https://github.com/praksharma/Stiff-PDEs-and-Physics-Informed-Neural-Networks/blob/main/1.%20Problem%201/11.%20Model%2011/plotting.ipynb
    
    data = np.load(f'{sweep_specific_directory}/inferencers/inf_data.npz',allow_pickle=True) # path verified in first example
    
    
    key = list(data.keys())
    dictin = data[key[0]]
    # how to access this numpy array?
    # there is a dictionary built inside the numpy array
    # https://stackoverflow.com/a/37949466/14598633
    data =  dictin[()]
    keys = list(data.keys())
    x_array, y_array, k_array, best_u_pred = data[keys[0]], data[keys[1]], data[keys[2]], data[keys[3]]
    # just an array to store x, y for griddata interpolation
    nodes_deepxde = np.hstack([x_array, y_array])
    
    # converting to absolute paths
    first_file = np.genfromtxt(to_absolute_path('true_solution/k=0.1_test.dat'))
    second_file = np.genfromtxt(to_absolute_path('true_solution/k=0.5_test.dat'))
    third_file = np.genfromtxt(to_absolute_path('true_solution/k=1.0_test.dat'))

    nodes_first = first_file[:,0:2] # deepxde nodes
    nodes_second = first_file[:,0:2] # deepxde nodes
    nodes_third = first_file[:,0:2] # deepxde nodes
    # solutions at the nodes
    first_true = first_file[:,2][:,None]
    second_true = second_file[:,2][:,None]
    third_true = third_file[:,2][:,None]
    
    ub = [x_array.max(), y_array.max()]
    lb = [x_array.min(), y_array.min()]
    
    # a function to search for index
    def return_indices(lower_limit_k, upper_limit_k, k):
        # returns indices for k (conductivity) between lower_limit_k and upper_limit_k
        indices = []
        for index,item in enumerate(k):
            if upper_limit_k>item>lower_limit_k:
                indices.append(index)
        # move out of the function to see the plot 
        #print(np.shape(indices))
        #plt.plot(k[indices])
        return indices
    
    
    fig, ax = plt.subplots(2, 3,dpi=200)
    #ax = fig.add_subplot(111)#, projection='3d')
    n_points = 1000
    # Intepolation points
    x = np.linspace(lb[0], ub[0], n_points)
    y = np.linspace(lb[1], ub[1], n_points)
    # Create meshgrid
    X, Y = np.meshgrid(x,y)
    # PINN predictions
    print(f'All possible values of conductivity :  {np.unique(k_array)}')
    indices = return_indices(lower_limit_k = 0.05, upper_limit_k=0.15, k=k_array) # k=0.4
    new_data = griddata(nodes_deepxde[indices], best_u_pred[indices].flatten(), (X, Y), method='linear')
    sc1 =  ax[0,0].imshow(new_data, interpolation='nearest', cmap=plt.get_cmap('plasma', 15), 
                      extent=[nodes_deepxde[:,0].min(), nodes_deepxde[:,0].max(), nodes_deepxde[:,1].min(), nodes_deepxde[:,1].max()], 
                      origin='lower', aspect='equal',vmin=0, vmax=1)
    #ax[0,0].set_ylim([0, 1.01])
    sc1.cmap.set_under('white')
    sc1.cmap.set_over('gray')

    #indices = return_indices(lower_limit_k = 0.58, upper_limit_k=0.62, k=k_array)
    new_data_1 = griddata(nodes_first, first_true.flatten(), (X, Y), method='linear')
    sc2 =  ax[1,0].imshow(abs(new_data_1-new_data), interpolation='nearest', cmap=plt.get_cmap('plasma', 15), 
                      extent=[nodes_deepxde[:,0].min(), nodes_deepxde[:,0].max(), nodes_deepxde[:,1].min(), nodes_deepxde[:,1].max()], 
                      origin='lower', aspect='equal',vmin=0, vmax=0.05)
    #ax[1,0].set_ylim([0, 1.01])
    sc2.cmap.set_under('white')
    sc2.cmap.set_over('gray')

    indices = return_indices(lower_limit_k = 0.45, upper_limit_k=0.55, k=k_array) # k=0.6
    new_data = griddata(nodes_deepxde[indices], best_u_pred[indices].flatten(), (X, Y), method='linear')
    sc1 =  ax[0,1].imshow(new_data, interpolation='nearest', cmap=plt.get_cmap('plasma', 15), 
                      extent=[nodes_deepxde[:,0].min(), nodes_deepxde[:,0].max(), nodes_deepxde[:,1].min(), nodes_deepxde[:,1].max()], 
                      origin='lower', aspect='equal',vmin=0, vmax=1)
    #ax[0,1].set_ylim([0, 1.03])
    sc1.cmap.set_under('white')
    sc1.cmap.set_over('gray')

    #indices = return_indices(lower_limit_k = 0.88, upper_limit_k=0.92, k=k_array)
    new_data_1 = griddata(nodes_second, second_true.flatten(), (X, Y), method='linear')
    sc2 =  ax[1,1].imshow(abs(new_data_1-new_data), interpolation='nearest', cmap=plt.get_cmap('plasma', 15), 
                      extent=[nodes_deepxde[:,0].min(), nodes_deepxde[:,0].max(), nodes_deepxde[:,1].min(), nodes_deepxde[:,1].max()], 
                      origin='lower', aspect='equal',vmin=0, vmax=0.05)
    #ax[1,1].set_ylim([0, 1.03])
    sc2.cmap.set_under('white')
    sc2.cmap.set_over('gray')

    indices = return_indices(lower_limit_k = 0.95, upper_limit_k=1.05, k=k_array) # k=0.8
    new_data = griddata(nodes_deepxde[indices], best_u_pred[indices].flatten(), (X, Y), method='linear')
    sc1 =  ax[0,2].imshow(new_data, interpolation='nearest', cmap=plt.get_cmap('plasma', 15), 
                      extent=[nodes_deepxde[:,0].min(), nodes_deepxde[:,0].max(), nodes_deepxde[:,1].min(), nodes_deepxde[:,1].max()], 
                      origin='lower', aspect='equal',vmin=0, vmax=1)
    #ax[0,2].set_ylim([0, 1.05])
    sc1.cmap.set_under('white')
    sc1.cmap.set_over('gray')

    #indices = return_indices(lower_limit_k = 0.96, upper_limit_k=1.0, k=k_array)
    new_data_1 = griddata(nodes_third, third_true.flatten(), (X, Y), method='linear')
    sc2 =  ax[1,2].imshow(abs(new_data_1-new_data), interpolation='nearest', cmap=plt.get_cmap('plasma', 15), 
                      extent=[nodes_deepxde[:,0].min(), nodes_deepxde[:,0].max(), nodes_deepxde[:,1].min(), nodes_deepxde[:,1].max()], 
                      origin='lower', aspect='equal',vmin=0, vmax=0.05)
    #ax[1,2].set_ylim([0, 1.05])
    sc2.cmap.set_under('white')
    sc2.cmap.set_over('gray')


    #sc = ax.scatter(nodes[:,0], nodes[:,1], c = best_u_pred, cmap=cm.jet, vmin=0, vmax=1)
    #plt.colorbar(sc)
    # setting color bounds on PINN predictions only
    sc1.cmap.set_under('white')
    sc1.cmap.set_over('gray')
    sc2.cmap.set_under('white')
    sc2.cmap.set_over('gray')

    #  setting axis labels and size
    ax[0,0].set(xlabel='x', ylabel='y')
    # big axis lebel looks ugly, defauly is 10, I set it to 8. see this: https://stackoverflow.com/a/46651121/14598633
    ax[0,0].xaxis.label.set_size(8)
    ax[0,0].yaxis.label.set_size(8)
    #ax[0,0].set_xticks([])
    ax[0,1].set(xlabel='x', ylabel='')
    ax[0,1].xaxis.label.set_size(8)
    ax[0,1].yaxis.label.set_size(8)
    #ax[0,1].set_xticks([])
    #ax[0,1].set_yticks([])
    ax[0,2].set(xlabel='x', ylabel='')
    ax[0,2].xaxis.label.set_size(8)
    ax[0,2].yaxis.label.set_size(8)
    #ax[0,2].set_xticks([])
    #ax[0,2].set_yticks([])
    ax[1,0].set(xlabel='x', ylabel='y')
    ax[1,0].xaxis.label.set_size(8)
    ax[1,0].yaxis.label.set_size(8)
    ax[1,1].set(xlabel='x', ylabel='')
    ax[1,1].xaxis.label.set_size(8)
    ax[1,1].yaxis.label.set_size(8)
    #ax[1,1].set_yticks([])
    ax[1,2].set(xlabel='x', ylabel='')
    ax[1,2].xaxis.label.set_size(8)
    ax[1,2].yaxis.label.set_size(8)
    #ax[1,2].set_yticks([])
    #ax[1,2].tick_params(axis='x', labelsize= 5)

    # Setting title
    ax[0,0].set_title('k = 0.1',fontsize=8)
    ax[0,1].set_title('k = 0.5',fontsize=8)
    ax[0,2].set_title('k = 1.0',fontsize=8)
    # ax[1,0].set_title('L=1.03',fontsize=8)
    # ax[1,1].set_title('L=1.04',fontsize=8)
    # ax[1,2].set_title('L=1.05',fontsize=8)

    # tight layout is mostly good but not here. Plots becomes too small to see anything.
    #fig.tight_layout()

    # see this for tick label [numbers on axis] adjustment: https://stackoverflow.com/a/11386056/14598633
    # default is 10
    for col in range(3):
        for row in range(2):
            ax[row, col].tick_params(axis='both', which='major', labelsize=8)


    # This one is better than tight_layout
    # adjust width and height:https://stackoverflow.com/a/6541454/14598633
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.25   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.5   # the amount of width reserved for blank space between subplots
    hspace = 0.1   # the amount of height reserved for white space between subplots

    plt.subplots_adjust(left=None, bottom=bottom, right=None, top=None, wspace=wspace, hspace=hspace)

    # for customised colormaps, see: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/colorbar_placement.html
    cbar1 = fig.colorbar(sc1, ax=[ax[0,0],ax[0,1],ax[0,2]], shrink=0.7, extend="both")
    cbar1.ax.tick_params(labelsize=8) # colorbar label size
    cbar2 = fig.colorbar(sc2, ax=[ax[1,0],ax[1,1],ax[1,2]], shrink=0.7,extend="both")
    cbar2.ax.tick_params(labelsize=8) # colorbar label size
    #It turns out cbar2 looks very ugly with only 2 ticks, so to add more tick/bins, do the following.
    tick_locator = ticker.MaxNLocator(nbins=4) # 6 bins comes from hit and trial. No rules
    cbar2.locator = tick_locator
    cbar2.update_ticks()    
    
    plt.savefig(f'{sweep_specific_directory}/solutions.jpg', dpi = 500,bbox_inches='tight',transparent=True)
    # https://stackoverflow.com/a/63076575/14598633
    plt.close() # release the plot data from RAM
    
    logging.info('Calculating relative L2 error')
    # Calculating the relative l2 error
    # k=0.1
    # k=0.4
    indices = return_indices(lower_limit_k = 0.05, upper_limit_k=0.15, k=k_array)
    new_data = griddata(nodes_deepxde[indices], best_u_pred[indices].flatten(), (X, Y), method='linear')
    new_data_1 = griddata(nodes_first, first_true.flatten(), (X, Y), method='linear')
    # to ignore NANs
    # https://datascience.stackexchange.com/a/11933

    new_data = np.nan_to_num(new_data)
    new_data_1 = np.nan_to_num(new_data_1)

    rel_l2_norm_1 = np.linalg.norm(abs(new_data-new_data_1),2)/np.linalg.norm(new_data_1,2)
    #print("Data 1:",rel_l2_norm_1)
    # K= 0.6
    indices = return_indices(lower_limit_k = 0.45, upper_limit_k=0.55, k=k_array)
    new_data = griddata(nodes_deepxde[indices], best_u_pred[indices].flatten(), (X, Y), method='linear')
    new_data_1 = griddata(nodes_second, second_true.flatten(), (X, Y), method='linear')

    new_data = np.nan_to_num(new_data)
    new_data_1 = np.nan_to_num(new_data_1)

    rel_l2_norm_2 = np.linalg.norm(abs(new_data-new_data_1),2)/np.linalg.norm(new_data_1,2)
    #print("Data 2:",rel_l2_norm_2)
    # k =0.8
    indices = return_indices(lower_limit_k = 0.95, upper_limit_k=1.05, k=k_array)
    new_data = griddata(nodes_deepxde[indices], best_u_pred[indices].flatten(), (X, Y), method='linear')
    new_data_1 = griddata(nodes_third, third_true.flatten(), (X, Y), method='linear')

    new_data = np.nan_to_num(new_data)
    new_data_1 = np.nan_to_num(new_data_1)

    rel_l2_norm_3 = np.linalg.norm(abs(new_data-new_data_1),2)/np.linalg.norm(new_data_1,2)
    #print("Data 3:",rel_l2_norm_3)

    average_L2_error = (rel_l2_norm_1 + rel_l2_norm_2 + rel_l2_norm_3)/3
    #print("average",average_L2_error)
        
    rel_l2_norm_of_error = average_L2_error # way around for less pain 
    
    logging.info(f'Relative L2 Error : {rel_l2_norm_of_error}')
    logging.info(f'Upperbound : {ub}')
    logging.info(f'Lowerbound : {lb}')
    
    
    # Store the stdout training output in a variable
    print_output = buffer.getvalue()
    # Store the training output
    logging.info('#################### Training data #######################')
    logging.info(print_output)
    
    logging.info('*******************TRAINING FINISHED*******************') 
    
    logging.info('*******************PLOTTING FINISHED*******************')

    # ML flow logger
    logging.info('*******************MLFLOW ROUTINE*******************')

    # If we don't change the $pwd the mlflow will store information inside the multirun folder for hydra 1.1.1. This shouldn't be done for Hydra 1.2 and above
    # unfortunately, the mlflow_ui will not those files.
    # we can also set the target location.
    
    python_path = str(Path(to_absolute_path("run.py")).resolve().parent) # absolute path of the python script # https://stackoverflow.com/a/46061872/14598633
    # returns the script path: '/scratch/s.1915438/nht_b/3. multirun/nht_b/3. multirun'
    
    
    logging.info(f'MlFlow URI location : {python_path}') # log the path for debugging
    
    mlflow.set_tracking_uri(f"file:{str(python_path)}/mlruns") # https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded
    with mlflow.start_run(): # start mlflow tracking
        # logging the parameters
        mlflow.log_param("Activation", cfg.activation)
        mlflow.log_param("lr", cfg.optimizer.lr)
        mlflow.log_param("Layers", cfg.layers)
        mlflow.log_param("Architecture", cfg.custom.arch)
        #mlflow.log_param("Iterations converged", iter_convergence)
        mlflow.log_param("Relative L2 error", rel_l2_norm_of_error)
        mlflow.log_param('Training time', elapsed)
        # Logging the final loss
        #mlflow.log_param('Training loss',total_loss_array[-1])
        #mlflow.log_param('PDE loss', PDE_loss_array[-1])
        #mlflow.log_param('BC loss', BC_loss_array[-1])
        # logging the saved figures
        #mlflow.log_artifact(sweep_dir + 'loss_plot.jpg')
        mlflow.log_artifact(f'{sweep_specific_directory}/solutions.jpg')
  
    logging.info('******************* END *******************')
    logging.shutdown()     # Shutdown the logger


# List of activations
#     ELU = enum.auto()
#     LEAKY_RELU = enum.auto()
#     MISH = enum.auto()
#     RELU = enum.auto()
#     GELU = enum.auto()
#     SELU = enum.auto()
#     PRELU = enum.auto()
#     SIGMOID = enum.auto()
#     SILU = enum.auto()
#     SIN = enum.auto()
#     SQUAREPLUS = enum.auto()
#     SOFTPLUS = enum.auto()
#     TANH = enum.auto()
#     STAN = enum.auto()
#     IDENTITY = enum.auto()
    
#print('Running NVIDIA Modulus simulation')

if __name__ == "__main__":
    run() # run modulus solver routine

