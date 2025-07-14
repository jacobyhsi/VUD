import sys
from typing import Optional
import os
import scienceplots
import matplotlib.pyplot as plt
import matplotlib as mpl

def modify_sys_path():
    """
    Run this function to add the src directory to sys.path.
    """
    # Get the absolute path of the project root
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

    # Add the src directory to sys.path
    sys.path.append(project_root)

    return

def get_parent_dir_path(path: Optional[str]):
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    
    if path is None:
        return project_root
    else:
        return os.path.join(project_root, path)
    
def set_plot_style():
    plt.style.use(['science','no-latex', 'retro', 'grid'])

    # Set font size for title
    mpl.rcParams['axes.titlesize'] = 18
    # Set font size for labels
    mpl.rcParams['axes.labelsize'] = 18
    # Set font size for tick labels
    mpl.rcParams['xtick.labelsize'] = 18
    mpl.rcParams['ytick.labelsize'] = 18
    mpl.rcParams['legend.fontsize'] = 14