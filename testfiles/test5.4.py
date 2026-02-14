import numpy as np
import pandas as pd
import os
from scipy.linalg import eigh

def run_test_5_4():
    # 1. Path Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, 'data', 'test5_3.csv')
    output_path = os.path.join(script_dir, 'testout_5.4_mk.csv')
    
    if not os.path.exists(input_path):
        print(f"Error: Could not find {input_path}")
        return

    # 2. Load Input
    df_in = pd.read_csv(input_path)
    headers = df_in.columns
    cov_in = df_in.values
    
    # 3. Near-PSD Fix (Ensuring the matrix is valid for PCA)
    vals, vecs = eigh(cov_in)
    vals[vals < 0] = 0
    cov_psd = vecs @ np.diag(vals) @ vecs.T
    
    # 4. PCA Analysis (99% Variance)
    # Re-calculate eigenvalues of the PSD matrix
    vals, vecs = eigh(cov_psd)
    
    # Sort eigenvalues and vectors descending
    idx = vals.argsort()[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    # Calculate how many components needed for 99%
    cum_var = np.cumsum(vals) / np.sum(vals)
    n_components = np.argmax(cum_var >= 0.99) + 1
    
    # 5. Simulation via PCA
    # L_pca = [Top Eigenvectors] * sqrt(Top Eigenvalues)
    L_pca = vecs[:, :n_components] @ np.diag(np.sqrt(vals[:n_components]))
    
    n_sims = 100000
    np.random.seed(42) # Seed for consistency
    
    # Generate random normal Z based on reduced dimensions
    Z = np.random.standard_normal((n_components, n_sims))
    
    # Project back to original space: X = L_pca * Z
    simulated_data = (L_pca @ Z).T
    
    # 6. Calculate Output Covariance
    cov_out = np.cov(simulated_data, rowvar=False)
    
    # 7. Save results
    df_out = pd.DataFrame(cov_out, columns=headers)
    df_out.to_csv(output_path, index=False)
    
    print("-" * 40)
    print(f"Simulation 5.4 (PCA 99%) Complete")
    print(f"Components used: {n_components} of {len(vals)}")
    print(f"Output saved: {output_path}")
    print("-" * 40)

if __name__ == "__main__":
    run_test_5_4()