###  Code Description

'''

NOTE: Uncomment and comment out the code below as needed.

'''

import subprocess
from morai_moe_code_forecasting import product_and_single_thread_testing

if __name__ == "__main__":
    try:
        

        ''' 
        # INFO: SINGLE STATE (Testing model for a single state)
        ''' 
        # product_and_single_thread_testing()

        ''' 
        # INFO: ALL STATES small/base (Testing model for all states)
        
        model: small or base
        
        ''' 
        
        processes = []

        model = "small"
        
        cmd = [
            "python", "run_all_morai_moe_script_5_years.py",
            model,
        ]
        p = subprocess.Popen(cmd)
        processes.append(p)
        
        for p in processes:
            p.wait()

    except Exception as e:
        print("An error occurred:", e)