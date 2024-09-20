import pandas as pd
import numpy as np
from pathlib import Path 

"""
Desing classes for doing meaningful gradient comparisons
Each class should represent a specific, controlled experiment
"""
class GradientComparisons():
    """
    Reads positional data (embeddings) from files and compute correct and incorrect gradients
    using them.

    Goal is to use the same underlying positions for gradient computation so we can really isolate
    how the 2 gradients result in different values. Otherwise, each embedding depends on the gradients,
    which then update to new embeddings, which the gradient is computed over etc...
    which creates different embeddings that might not result in meaningful gradient comparisons
    """
    def __init__(self, embeddings_folder_path, cf, key):
        self.folder_path = Path(embeddings_folder_path)
        self.cf = cf                # Cost function object (initialized)
        self.key = key              # Key to uniquely identify this experiment (used for saving results)
    
    def compute_gradients(self, V, grad_fix, output_folder_path):        
        # There may be multiple optimization steps, resulting in multiple folders
        for folder in self.folder_path.iterdir():
            # Loop over the different optimizer outputs
            for idx, embedding_file_path in enumerate(folder.iterdir()):
                # Read file content as embedding (Y)
                # Y - (n_samples, n_dim) 
                Y = np.loadtxt(str(embedding_file_path), delimiter=',')

                # Compute gradient per file
                error, grad = self.cf.obj_grad(Y, V=V, grad_fix=grad_fix)
                grad = grad.reshape(V.shape[0], self.cf.n_components)

                # Save gradients
                Path(f"{output_folder_path}/correct_grad:{grad_fix}").mkdir(parents=True, exist_ok=True)
                pd.DataFrame(grad).to_csv(f"{output_folder_path}/correct_grad:{grad_fix}/it:{idx - 1}.csv", header=False, index=False)

