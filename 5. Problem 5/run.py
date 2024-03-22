# Necessary packages
import os # execute terminal commands
import numpy as np
#import torch # device assignment for importance model adn GPU number assignment in Hydra zen Joblib
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

import matplotlib.pyplot as plt
from scipy.interpolate import griddata # interpolation for plt.imshow()
#from mpl_toolkits.mplot3d import Axes3D # for 3D if needed
# from matplotlib import cm # colormap
from matplotlib import ticker # controls number of ticks in colorbar, helpful for very ugly unsymmetric colorbar

# Logging things
import logging # inbuilt text logging
import mlflow # sweep specific logger with csv file and multimedia
# Store print stdout in the python logger
# https://bobbyhadz.com/blog/python-assign-string-output-to-variable
from io import StringIO
import sys



# The 2D steady state Diffusion PDE class
class DiffusionEquation2D(PDES):
    """
    2D steady state Diffusion PDE
    """

    name = "DiffusionEquation2D"
    def __init__(self, u="u", nu = "nu"):
        # coordinates
        x = Symbol("x")
        y = Symbol("y")
        
        self.u = u # extract the string before manupulating the value of u 
        # make input variables
        input_variables = {"x": x, "y": y}
        
        c = Number(nu)
        
        # Temperature output
        u = Function(u)(*input_variables)

        # set equations
        self.equations = {}
        self.equations["heat_equation_" + self.u] = (c * u.diff(x, 1)).diff(x,1) + (c * u.diff(y, 1)).diff(y,1)  # diff(variable,order of derivative)
        
# Implementation of the interface condition
class DiffusionInterface(PDES):
    """
    DiffusionInterface : Implementation of the interface condition
    """

    name = "DiffusionInterface"
    def __init__(self, u_1, u_2, nu_1, nu_2):
        
        # extract the string before manupulating the value of u 
        self.u_1 = u_1
        self.u_2 = u_2
        
        # coordinates
        x = Symbol("x")
        y = Symbol("y")
    
        normal_x, normal_y, normal_z = (Symbol("normal_x"),Symbol("normal_y"), Symbol("normal_z"))
        # make input variables
        input_variables = {"x": x, "y": y}
        
        c_1 = Number(nu_1)
        c_2 = Number(nu_2)
        
        # Temperature output
        u_1 = Function(u_1)(*input_variables)
        u_2 = Function(u_2)(*input_variables)
        
        flux_1 = c_1 * (normal_x * u_1.diff(x) + normal_y * u_1.diff(y))
        flux_2 = c_2 * (normal_x * u_2.diff(x) + normal_y * u_2.diff(y))
        
        # set equations
        self.equations = {}
        self.equations["diffusion_interface_dirichlet_" +  self.u_1 + "_" + self.u_2] = u_1 - u_2
        self.equations["diffusion_interface_neumann_" + self.u_1 + "_" + self.u_2] = (flux_1 - flux_2)
        
        
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
    # Logger for a specific sweep saved to f'{sweep_specific_directory}/training.log'
    logging.basicConfig(level=logging.INFO, filemode = 'w', format='%(asctime)s :: %(message)s', filename=f'{sweep_specific_directory}/training.log',force=True) # https://stackoverflow.com/a/71625766/14598633
    logging.info(f'Logger Initialised')
    #logging.info(f'Sweep number : {sweep_number}')
    logging.info(f'Sweep directory : {sweep_specific_directory}')
    
    logging.info(f'Learning Rate : {cfg.optimizer.lr}') # this is applied automatically
    logging.info(f'Activation : {cfg.activation}') # this one is not supported so doing it manually
    logging.info(f'Number of Layers : {cfg.layers}') # number of DGM layers

    
    # loading activations based on the hydra config
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
    # define nu 
    nu_1 = 1/30
    nu_2 = 1/15
    
    logging.info(f"nu1 = {nu_1}, nu2: {nu_2}")
    # PDE for both sides
    heat_u1 = DiffusionEquation2D(u="u_1", nu = nu_1)
    heat_u2 = DiffusionEquation2D(u="u_2", nu = nu_2)
    heat_in = DiffusionInterface("u_1", "u_2", nu_1, nu_2)
    
    input_keys=[Key("x"), Key("y")]
    output_keys_1=[Key("u_1")]
    output_keys_2=[Key("u_2")]
    
    logging.info('Detecting architecture')
    # make list of nodes to unroll graph on

    if cfg.custom.arch == "FullyConnectedArch":
        logging.info('Architecture: Fully Connected Arch')
        heat_net_1 = FullyConnectedArch(
            input_keys=input_keys,
            output_keys=output_keys_1,
            layer_size=cfg.layers,
            activation_fn=act_func, # set activation function
            adaptive_activations=cfg.custom.adaptive_activations,
        )
        heat_net_2 = FullyConnectedArch(
            input_keys=input_keys,
            output_keys=output_keys_2,
            layer_size=cfg.layers,
            activation_fn=act_func, # set activation function
            adaptive_activations=cfg.custom.adaptive_activations,
        )
    elif cfg.custom.arch == "FourierNetArch": # not used in this study
        logging.info('Architecture: Fourier Network')
        heat_net_1 = FourierNetArch(
            input_keys=input_keys,
            output_keys=output_keys_1,
            adaptive_activations=cfg.custom.adaptive_activations,
            activation_fn=act_func, # set activation function
            frequencies=("axis", [i for i in range(1)]),
            frequencies_params=("axis", [i for i in range(1)]),
        )
        heat_net_2 = FourierNetArch(
            input_keys=input_keys,
            output_keys=output_keys_2,
            adaptive_activations=cfg.custom.adaptive_activations,
            activation_fn=act_func, # set activation function
            frequencies=("axis", [i for i in range(1)]),
            frequencies_params=("axis", [i for i in range(1)]),
        )
    elif cfg.custom.arch == "SirenArch": # not used in this study
        logging.info('Architecture: SIREN Arch')
        heat_net = SirenArch(
            input_keys=input_keys,
            layer_size=cfg.layers,
            output_keys=output_keys,
            activation_fn=act_func, # set activation function
        )
    elif cfg.custom.arch == "ModifiedFourierNetArch":
        logging.info('Architecture: Modified Fourier Arch')
        heat_net_1 = ModifiedFourierNetArch(
            input_keys=input_keys,
            output_keys=output_keys_1,
            layer_size=cfg.layers,
            nr_layers=2,
            adaptive_activations=cfg.custom.adaptive_activations,
            activation_fn=act_func, # set activation function
            frequencies=("axis", [i for i in range(1)]),
            frequencies_params=("axis", [i for i in range(1)]),
        )
        heat_net_2 = ModifiedFourierNetArch(
            input_keys=input_keys,
            output_keys=output_keys_2,
            layer_size=cfg.layers,
            nr_layers=2,
            adaptive_activations=cfg.custom.adaptive_activations,
            activation_fn=act_func, # set activation function
            frequencies=("axis", [i for i in range(1)]),
            frequencies_params=("axis", [i for i in range(1)]),
        )
    elif cfg.custom.arch == "DGMArch":
        logging.info('Architecture: DGM Arch')
        heat_net_1 = DGMArch(
            input_keys=input_keys,
            output_keys=output_keys_1,
            layer_size=cfg.layers,
            nr_layers=1,
            adaptive_activations=cfg.custom.adaptive_activations,
            activation_fn=act_func, # set activation function
        )
        heat_net_2 = DGMArch(
            input_keys=input_keys,
            output_keys=output_keys_2,
            layer_size=cfg.layers,
            nr_layers=1,
            adaptive_activations=cfg.custom.adaptive_activations,
            activation_fn=act_func, # set activation function
        )
    else:
        sys.exit(
            "Network not configured for this script. Please include the network in the script"
        )
    
    nodes = (
        heat_u1.make_nodes()
        + heat_u2.make_nodes()
        + heat_in.make_nodes()
        + [heat_net_1.make_node(name="u1_network", jit=cfg.jit)]
        + [heat_net_2.make_node(name="u2_network", jit=cfg.jit)]
    )
    
    logging.info('Nodes Initialised')
        
    # make geometry
    x, y, u_1, u_2 = Symbol("x"), Symbol("y"), Symbol("u_1"), Symbol("u_2")
    lower_bound = (0, 0)
    interface_bound_1 = (0.5, 1)
    interface_bound_2 = (0.5, 0)
    upper_bound = (1, 1)

    rec_1 = Rectangle(lower_bound, interface_bound_1) # (x_1,y_1), (x_2,y_2)
    rec_2 = Rectangle(interface_bound_2, upper_bound) # (x_3,y_3), (x_3,y_4)

    logging.info('Geometry created')
    
    # make domain
    heat_domain = Domain()
    logging.info('Domain created')
    
    # Coordinates across the interface for the PDE bounds
    x_bound_1 = (0, 0.5)
    y_bound_1 = (0, 0.5)

    x_bound_2 = (0.5, 1.0)
    y_bound_2 = (0.5, 1.0)

    # Adding constraints
    # interior
    interior_u1 = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec_1,
        outvar={"heat_equation_u_1" : 0},
        batch_size=cfg.batch_size.Interior,
        bounds={x: x_bound_1, y: y_bound_1},
        # lambda weights for now
        # lambda_weighting={"heat_equation": 1 / (1 + exp(-100 * Abs(x - 0.5)))},
        #importance_measure=importance_measure,
        quasirandom=cfg.custom.quasirandom,
    )
    heat_domain.add_constraint(interior_u1, "interior_u_1")
    
    logging.info('Interface 1: Interior points created')
    
    interior_u2 = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec_2,
        outvar={"heat_equation_u_2" : 0},
        batch_size=cfg.batch_size.Interior,
        bounds={x: x_bound_2, y: y_bound_2},
        # lambda weights for now
        # lambda_weighting={"heat_equation": 1 / (1 + exp(-100 * Abs(x - 0.5)))},
        # importance_measure=importance_measure,
        quasirandom=cfg.custom.quasirandom,
    )
    heat_domain.add_constraint(interior_u2, "interior_u_2")
    
    logging.info('Interface 2: Interior points created')
    
    # Left wall
    left_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec_1,
        #outvar={"u_1": 0.07547169811320754},#2*(1/30) * u.diff(y,1)}, # Left Robin BC
        outvar={"u_1": 2*(1/30) * u_1.diff(y,1)}, # Left Robin BC
        batch_size=cfg.batch_size.Wall,
        #lambda_weighting={"T": 1.0 - 2 * Abs(y-0.5)},  # weight edges to be zero
        criteria=Eq(x, 0.0), # coordinates for sampling
        #importance_measure=importance_measure,
        quasirandom=cfg.custom.quasirandom,
    )
    heat_domain.add_constraint(left_wall, "left_wall")

    logging.info('Left wall BC created')
    
    # right wall
    right_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec_2,
        #outvar={"u_2": 0.9245283018867925},#1 - 2*(1/15) * u.diff(y,1)}, # Right Robin BC
        outvar={"u_2": 1 - 2*(1/15) * u_2.diff(y,1)}, # Right Robin BC
        batch_size=cfg.batch_size.Wall,
        #lambda_weighting={"T": 1.0 - 2 * Abs(y-0.5)},  # weight edges to be zero
        criteria=Eq(x, 1.0), # coordinates for sampling
        #importance_measure=importance_measure,
        quasirandom=cfg.custom.quasirandom,
    )
    heat_domain.add_constraint(right_wall, "right_wall")

    logging.info('Right wall BC created')
    
    # interface wall
    interface_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec_1,
        outvar={"diffusion_interface_dirichlet_u_1_u_2": 0,
               "diffusion_interface_neumann_u_1_u_2": 0}, # Interface consistencey
        batch_size=cfg.batch_size.Wall,
        #lambda_weighting={"T": 1.0 - 2 * Abs(y-0.5)},  # weight edges to be zero
        criteria=Eq(x, 0.5), # coordinates for sampling
        #importance_measure=importance_measure,
        quasirandom=cfg.custom.quasirandom,
    )
    heat_domain.add_constraint(interface_wall, "Interface_wall")

    logging.info('Interface 1: Interface wall created')
    
    interface_wall_1 = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec_2,
        #outvar={"u": 0.9245283018867925},#1 - 2*(1/15) * u.diff(y,1)}, # Right Robin BC
        outvar={"diffusion_interface_dirichlet_u_1_u_2": 0,
               "diffusion_interface_neumann_u_1_u_2": 0}, # Interface consistencey
        batch_size=cfg.batch_size.Wall,
        #lambda_weighting={"T": 1.0 - 2 * Abs(y-0.5)},  # weight edges to be zero
        criteria=Eq(x, 0.5), # coordinates for sampling
        #importance_measure=importance_measure,
        quasirandom=cfg.custom.quasirandom,
    )
    heat_domain.add_constraint(interface_wall_1, "Interface_wall_1")


    logging.info('Interface 2: Interface wall created')
    
    
    # Getting path (nvidia modulus only accepts absolute path to remove any ambiguity)
    validation_path = to_absolute_path("final_data.npz")
    data = np.load(validation_path) # Load data from file
    keys = list(data.keys()) # all keys in the dictionary
    logging.info(f'Validation dataset keys : {keys}')
    
    # loading data from the dictionary
    xy_pairs_1 = data[keys[0]]
    xy_pairs_2 = data[keys[1]]
    u_1 = data[keys[2]]
    u_2 = data[keys[3]]
    
    
    # This is the required format for validation data
    # Left domain
    openfoam_invar_numpy = {}
    openfoam_invar_numpy["x"] = xy_pairs_1[:, 0][:, None]
    openfoam_invar_numpy["y"] = xy_pairs_1[:, 1][:, None]

    openfoam_outvar_numpy = {}
    openfoam_outvar_numpy["u_1"] = u_1[:, None]

    openfoam_validator_1 = PointwiseValidator(
        openfoam_invar_numpy,
        openfoam_outvar_numpy,
        nodes,
        plotter=ValidatorPlotter(),
    )
    heat_domain.add_validator(openfoam_validator_1, name="Val1")

    # Right domain
    openfoam_invar_numpy = {}
    openfoam_invar_numpy["x"] = xy_pairs_2[:, 0][:, None]
    openfoam_invar_numpy["y"] = xy_pairs_2[:, 1][:, None]

    openfoam_outvar_numpy = {}
    openfoam_outvar_numpy["u_2"] = u_2[:, None]

    openfoam_validator_2 = PointwiseValidator(
        openfoam_invar_numpy,
        openfoam_outvar_numpy,
        nodes,
        plotter=ValidatorPlotter(),
    )
    heat_domain.add_validator(openfoam_validator_2, name="Val2")
    
    logging.info('Validation Data loaded')
    
    
    logging.info('Inference wasn\'t  Implemented')
    
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
    data = np.load(f'{sweep_specific_directory}/validators/Val1.npz',allow_pickle=True)
    key = list(data.keys())
    dictin = data[key[0]]
    data =  dictin[()]
    keys = list(data.keys())
    
    x_1, y_1, u_true_1, u_pred_1 = data[keys[0]], data[keys[1]], data[keys[2]], data[keys[3]]#, data[keys[4]]
    
    data = np.load(f'{sweep_specific_directory}/validators/Val2.npz',allow_pickle=True)
    key = list(data.keys())
    dictin = data[key[0]]
    data =  dictin[()]
    keys = list(data.keys())
    
    x_2, y_2, u_true_2, u_pred_2 = data[keys[0]], data[keys[1]], data[keys[2]], data[keys[3]]#, data[keys[4]]
    
    # combining the results across both the interfaces
    # combine both sides
    x_array = np.vstack([x_1, x_2])
    y_array = np.vstack([y_1, y_2])
    u_true = np.vstack([u_true_1, u_true_2])
    u_pred = np.vstack([u_pred_1, u_pred_2])
    
    ub = [x_array.max(), y_array.max()]#, z_array.max()]
    lb = [x_array.min(), y_array.min()]#, z_array.min()]
    
    # calculate the relative l2 error
    rel_l2_norm_of_error = np.linalg.norm(u_true-u_pred,2)/np.linalg.norm(u_true,2)
    logging.info(f'Relative L2 Error : {rel_l2_norm_of_error}')
    logging.info(f'Upperbound : {ub}')
    logging.info(f'Lowerbound : {lb}')    
    
    # Setting the meshgrid for plt.imshow()
    nodes = np.hstack([x_array, y_array])
    n_points = 300
    # Intepolation points
    x = np.linspace(lb[0], ub[0], n_points)
    y = np.linspace(lb[1], ub[1], n_points)
    # Create meshgrid
    X, Y = np.meshgrid(x,y)
    
    # Plotting
    fig, ax = plt.subplots(1, 2,dpi=300)

    # deepxde result
    data_deepxde = griddata(nodes, u_pred.flatten(), (X, Y), method='linear')
    sc1 = ax[0].imshow(data_deepxde, interpolation='nearest', cmap=plt.get_cmap('plasma', 10), 
                      extent=[nodes[:,0].min(), nodes[:,0].max(), nodes[:,1].min(), nodes[:,1].max()], 
                      origin='lower', aspect='equal')#,vmin=0, vmax=1)

    n_points = 300
    # Intepolation points
    x = np.linspace(lb[0], ub[0], n_points)
    y = np.linspace(lb[1], ub[1], n_points)
    # Create meshgrid
    X, Y = np.meshgrid(x,y)

    # absolute pointwise difference
    data_difference = griddata(nodes, u_true.flatten(), (X, Y), method='linear')
    sc2 = ax[1].imshow(abs(data_difference-data_deepxde), interpolation='nearest', cmap=plt.get_cmap('plasma', 7), 
                      extent=[nodes[:,0].min(), nodes[:,0].max(), nodes[:,1].min(), nodes[:,1].max()], 
                      origin='lower', aspect='equal',vmin=0, vmax=0.05)

    sc1.cmap.set_under('white')
    sc1.cmap.set_over('gray')
    sc2.cmap.set_under('white')
    sc2.cmap.set_over('gray')

    #sc = ax.scatter(nodes[:,0], nodes[:,1], c = temperature,s=3, cmap=cm.jet)
    #plt.colorbar(sc)
    ax[0].set(xlabel='x', ylabel='y')
    ax[1].set(xlabel='x', ylabel='')
    #fig.tight_layout()
    # This one is better than tight_layout
    # adjust width and height:https://stackoverflow.com/a/6541454/14598633
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.3   # the amount of width reserved for blank space between subplots
    hspace = 0.1   # the amount of height reserved for white space between subplots

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=None)

    #fig.colorbar(sc1, ax=ax.ravel().tolist(),shrink=0.5)
    fig.colorbar(sc1, ax=ax[0],shrink=0.5,extend='both', pad=0.1)
    fig.colorbar(sc2, ax=ax[1],shrink=0.5,extend='both', pad=0.1)

    #plt.title('Temperature distribution')
    plt.savefig(f'{sweep_specific_directory}/solutions.jpg', dpi = 500,bbox_inches='tight',transparent=True)
    # https://stackoverflow.com/a/63076575/14598633
    plt.close() # release the plot data from RAM
    

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
    # unfortunately, the mlflow_ui will not recognise those files.
    # we can also set the target location.
    
    python_path = str(Path(str(validation_path)).resolve().parent) # absolute path of the python script # https://stackoverflow.com/a/46061872/14598633
    # returns the script path: '/scratch/s.1915438/nht_b/3. multirun/nht_b/3. multirun' or similar thing
    
    
    logging.info(f'MlFlow URI location : {python_path}') # log the path for debugging
    
    mlflow.set_tracking_uri(f"file:{str(python_path)}/mlruns") # https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded
    with mlflow.start_run(): # start mlflow tracking
        # logging the parameters
        mlflow.log_param("Activation", cfg.activation)
        mlflow.log_param("lr", cfg.optimizer.lr)
        mlflow.log_param("Layers", cfg.layers)
        mlflow.log_param("Architecture", cfg.custom.arch)
        mlflow.log_param("Relative L2 error", rel_l2_norm_of_error)
        mlflow.log_param('Training time', elapsed)
        # Logging the final loss
        #mlflow.log_param('Training loss',total_loss_array[-1])
        #mlflow.log_param('PDE loss', PDE_loss_array[-1])
        #mlflow.log_param('BC loss', BC_loss_array[-1])
        # logging the saved figures
        #mlflow.log_artifact(sweep_dir + 'loss_plot.jpg')
        mlflow.log_artifact(f'{sweep_specific_directory}/solutions.jpg')
  
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

