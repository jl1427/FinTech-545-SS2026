import numpy as np
import pandas as pd
import os

def run_test_5_1():
    # 1. Path Configuration
    # Finds the 'testfiles' folder where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Points to 'testfiles/data/test5_1.csv'
    input_path = os.path.join(script_dir, 'data', 'test5_1.csv')
    # Saves output in 'testfiles/testout_5.1_mk.csv'
    output_path = os.path.join(script_dir, 'testout_5.1_mk.csv')
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Could not find the file at {input_path}")
        print("Please verify that 'test5_1.csv' is inside the 'data' folder.")
        return

    # 2. Load Input Covariance Matrix
    df_in = pd.read_csv(input_path)
    headers = df_in.columns
    cov_in = df_in.values
    
    # 3. Simulation Parameters
    n_sims = 100000
    mean = np.zeros(cov_in.shape[0])
    
    # Set a random seed for reproducibility
    # Note: Different seeds/numpy versions may produce slight variations in decimals
    np.random.seed(0) 
    
    # 4. Perform Normal Simulation
    # multivariate_normal creates 100,000 samples that follow the input covariance
    simulated_data = np.random.multivariate_normal(mean, cov_in, size=n_sims)
    
    # 5. Calculate Empirical Covariance
    # We use rowvar=False because the assets (x1...x5) are in columns
    cov_out = np.cov(simulated_data, rowvar=False)
    
    # 6. Save results to CSV
    df_out = pd.DataFrame(cov_out, columns=headers)
    df_out.to_csv(output_path, index=False)
    
    print("-" * 40)
    print("Simulation 5.1 Successful")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("-" * 40)

if __name__ == "__main__":
    run_test_5_1()