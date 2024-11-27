import os
import csv
import json
import numpy as np
import pandas as pd
from hyperbolicTSNE.visualization import plot_poincare, animate
from pathlib import Path


def find_last_embedding(log_path):
    """ Give a path with logging results, finds the last embedding saved there.
    """
    for subdir, dirs, files in reversed(list(os.walk(log_path))):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']

        for fi, file in enumerate(reversed(sorted(files, key=lambda x: int(x.split(".")[0])))):
            root, ext = os.path.splitext(file)
            if ext == ".csv":
                total_file = subdir.replace("\\", "/") + "/" + file

                return np.genfromtxt(total_file, delimiter=',')


def find_ith_embedding(log_path, i):
    """ Give a path with logging results, finds the i-th embedding saved there.
    """
    j = 0
    for subdir, dirs, files in list(os.walk(log_path)):
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']

        for fi, file in enumerate(sorted(files, key=lambda x: int(x.split(", ")[0]))):
            root, ext = os.path.splitext(file)
            if ext == ".csv":
                j += 1

                if j >= i:
                    total_file = subdir.replace("\\", "/") + "/" + file

                    return np.genfromtxt(total_file, delimiter=',')
            

def opt_config(cf, learning_rate, exaggeration_factor, ex_iterations, main_iterations, 
               exact, vanilla=True, grad_scale_fix=True, grad_fix=True, size_tol=0.999, 
               max_no_progress=2000, min_grad_norm=1e-10):
    """
    Return an opt_config dict with all the parameters set
    """
    return dict(
        cf=cf, # Cost function to use
        learning_rate_ex=learning_rate,  # learning rate during exaggeration
        learning_rate_main=learning_rate,  # learning rate main optimization 
        exaggeration=exaggeration_factor, 
        exaggeration_its=ex_iterations, 
        gradientDescent_its=main_iterations, 
        vanilla=vanilla,  # if vanilla is set to true, regular gradient descent without any modifications is performed; for  vanilla set to false, the optimization makes use of momentum and gains
        momentum_ex=0.5,  # Set momentum during early exaggeration to 0.5
        momentum=0.2,  # Set momentum during non-exaggerated gradient descent to 0.8
        exact=exact,  # To use the quad tree for acceleration (like Barnes-Hut in the Euclidean setting) or to evaluate the gradient exactly
        area_split=False,  # To build or not build the polar quad tree based on equal area splitting or - alternatively - on equal length splitting
        n_iter_check=2000,  # Needed for early stopping criterion
        size_tol=size_tol,  # Size of the embedding to be used as early stopping criterion
        grad_scale_fix=grad_scale_fix,
        grad_fix=grad_fix,
        n_iter_without_progress=max_no_progress, # Max. nr. of iterations with no progress (worse gradients) before we stop
        min_grad_norm=min_grad_norm, # Min. norm of gradient we accept before aborting
    )   


def initialize_logger(opt_params, opt_config, log_path=None, grad_path=None, only_animate=False):
    """
    Sets the correct values for the logger keys in the opt_params, opt_config
    dictionaries.
    """
    logging_dict = {
        "log_path": log_path,
        "log_grad_path": grad_path,
    }

    # print(opt_params["logging_dict"])
    opt_params["logging_dict"] = logging_dict

    log_path = opt_params["logging_dict"]["log_path"]
    # Delete old log path
    if os.path.exists(log_path) and not only_animate:
        import shutil
        shutil.rmtree(log_path)
    # End: logging

    if os.path.exists(grad_path) and not only_animate:
        import shutil 
        shutil.rmtree(grad_path)

    print(f"config: {opt_config}")    
    
    return opt_params, opt_config


def write_data(data_header, data_row, file_path="results/results.csv"):
    """
    Function that writes data to a csv file

    data_row = [self.dataset, self.num_points, 
            self.dataX.shape[1], self.pca_components, 
            self.perp, self.htsne.cf, 
            self.htsne.runtime, self.htsne.its,
            self.opt_config['exact'], self.correct_gradient]
    
    data_header = ['dataset', 'data_size', 
                        'data_dim', 'pca_init', 
                        'perplexity', 'cost_function_value', 
                        'runtime', 'total_iterations', 
                        'exact', 'correct gradient']
    """
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)

        # Create a header if file doesnt exist
        if not file_exists:
            writer.writerow(data_header)

        # Store experiment results
        writer.writerow(data_row)














#################################
# Functions for storing results #
#################################

def store_visuals(hyperbolicEmbedding, dataLabels, save_folder, file_name, opt_params):
    """
    Create and store visualizations (.png and .gif) of our embedding
    """
    # Create folder if it doesn't exist
    path = Path(save_folder)
    path.mkdir(parents=True, exist_ok=True)

    fig = plot_poincare(hyperbolicEmbedding, dataLabels)
    fig.savefig(f"{file_name}.png")

    # Save animation
    animate(opt_params["logging_dict"], dataLabels, f"{file_name}.gif", fast=True, plot_ee=True)


def save_experiment_results(save_folder, data_fig, emb_fig, opt_params, dataLabels, exp_data, embedding, htsne, step=10):
    """
    save_folder:        general folder to save results to
    data_fig:           original data pyplot figure
    emb_fig:            embedding data pyplot figure
    opt_params:
    dataLabels:
    exp_data:
    """
    # Create folder for storing results
    Path(f"{save_folder}").mkdir(parents=True, exist_ok=True)

    # Create data plot & save
    if data_fig is not None:
        data_fig.savefig(save_folder + "/data.png")
        data_fig.tight_layout()

    # Create emb. plot & save
    if emb_fig is not None:
        emb_fig.savefig(save_folder+ "/emb.png")
        emb_fig.tight_layout()

    # Create emb. animation & save
    file_name = save_folder + "/emb_anim.gif"
    animate(opt_params["logging_dict"], dataLabels, file_name, fast=True, plot_ee=True, step=step, size=1)

    # Store experiment data in csv
    with open(save_folder + "data.json", 'w') as f:
        json.dump(exp_data, f)

    # Store most recent embedding into csv
    pd.DataFrame(embedding).to_csv(f"{save_folder}/emb.csv", header=False, index=False)

    # Store cost function values
    file_name = f"{save_folder}/cfs.txt"
    with open(file_name, 'w') as f:
        for cf_val in htsne.optimizer.cf.cf_values:
            f.write(f"{cf_val}\n")



def next_experiment_folder_id(folder):
    """ 
    folder:     path to folder containing experiment folders named as 
                'experiment_{id}'

    Returns the id of the next experiment folder we want to save experiments to by looking
    at existing experiment folders' id's and returning the largest one + 1 there.
    """
    folder = Path(folder)
    max_fdr_nr = 0
    for item in folder.iterdir():
        if item.is_dir():
            idx = int(str(item).split('_')[-1])
            if idx > max_fdr_nr:
                max_fdr_nr = idx

    return max_fdr_nr + 1



def GaussianKL_Tree_results(n_children, depth, cluster_size, dist, n_nodes, htsne, ex_iterations,
                            main_iterations, lr, cf, hyp_var, size_tol, max_dist_H, max_dist, correct_grad,
                            grad_scale_fix, exact_grad, exaggeration_factor, optim_procedure, description=None):    
    return {
            "dataset":     "Tree_Dataset",
            'dataset_info:':        {
                    'n_children': n_children,
                    'depth': depth,
                    'cluster_size': cluster_size,
                    'dist': dist,
                    'n_nodes': int(n_nodes),
                },
            "run_time:":            htsne.runtime,
            "ex_iterations":        ex_iterations,
            "main_iterations":      main_iterations, 
            "actual_iterations":    htsne.its,
            "learning_rate":        lr,
            "cost_function":        cf.class_str(), 
            "variance_info": {                              
                "var":              hyp_var,
                "size_tol":         size_tol,
                "max_dist_H":       max_dist_H,
                "max_dist":         max_dist,
            },
            "correct_grad":         correct_grad,
            "scale_fix":            grad_scale_fix,
            "exact":                exact_grad,
            "exaggeration_factor":  0 if ex_iterations == 0 else exaggeration_factor,
            "optim_procedure":      optim_procedure,   
            "extra info":           description,   
        }

def GaussianKL_results(name, htsne, ex_iterations, lr, perp, num_p, pca,
                       main_iterations, cf, hyp_var, size_tol, max_dist_H, max_dist, correct_grad,
                       grad_scale_fix, exact_grad, exaggeration_factor, optim_procedure, description=None):    
    return {
            "dataset":     name,
            'dataset_info': {
                'perplexity': perp, 
                'num_points': num_p,
                'pca':        pca,
            },
            "run_time:":            htsne.runtime,
            "ex_iterations":        ex_iterations,
            "main_iterations":      main_iterations, 
            "actual_iterations":    htsne.its,
            "learning_rate":        lr,
            "cost_function":        cf.class_str(), 
            "variance_info": {                              
                "var":              hyp_var,
                "size_tol":         size_tol,
                "max_dist_H":       max_dist_H,
                "max_dist":         max_dist,
            },
            "correct_grad":         correct_grad,
            "scale_fix":            grad_scale_fix,
            "exact":                exact_grad,
            "exaggeration_factor":  0 if ex_iterations == 0 else exaggeration_factor,
            "optim_procedure":      optim_procedure,   
            "extra info":           description,   
        }

def HyperbolicKL_Tree_results(n_children, depth, cluster_size, dist, n_nodes, htsne, ex_iterations,
                              main_iterations, lr, cf, correct_grad,
                              grad_scale_fix, exact_grad, exaggeration_factor, optim_procedure, description=None):
    return {
            "dataset":     "Tree_Dataset",
            'dataset_info:':        {
                    'n_children': n_children,
                    'depth': depth,
                    'cluster_size': cluster_size,
                    'dist': dist,
                    'n_nodes': int(n_nodes),
                },
            "run_time:":            htsne.runtime,
            "ex_iterations":        ex_iterations,
            "main_iterations":      main_iterations, 
            "actual_iterations":    htsne.its,
            "learning_rate":        lr,
            "cost_function":        cf.class_str(), 
            "correct_grad":         correct_grad,
            "scale_fix":            grad_scale_fix,
            "exact":                exact_grad,
            "exaggeration_factor":  0 if ex_iterations == 0 else exaggeration_factor,
            "optim_procedure":      optim_procedure,   
            "extra info":           description,   
    }


def HyperbolicKL_results(name, htsne, ex_iterations, lr, perp, num_p, pca,
                       main_iterations, cf, correct_grad,
                       grad_scale_fix, exact_grad, exaggeration_factor, optim_procedure, description=None):    
    return {
            "dataset":     name,
            'dataset_info': {
                'perplexity': perp, 
                'num_points': num_p,
                'pca':        pca,
            },
            "run_time:":            htsne.runtime,
            "ex_iterations":        ex_iterations,
            "main_iterations":      main_iterations, 
            "actual_iterations":    htsne.its,
            "learning_rate":        lr,
            "cost_function":        cf.class_str(), 
            "correct_grad":         correct_grad,
            "scale_fix":            grad_scale_fix,
            "exact":                exact_grad,
            "exaggeration_factor":  0 if ex_iterations == 0 else exaggeration_factor,
            "optim_procedure":      optim_procedure,   
            "extra info":           description,   
        }