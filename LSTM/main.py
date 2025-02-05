import subprocess
import multiprocessing


if __name__ == "__main__":
    try:

        ''' 
        # INFO: TEST PYTORCH
        ''' 
        # from long_short_term_memory_pytorch import product_and_single_thread_testing
        # multiprocessing.set_start_method("spawn")
        # thread = multiprocessing.Process(target=product_and_single_thread_testing)
        # thread.start()
        # thread.join()

        ''' 
        # INFO: ALL STATES RECURSIVE/DIRECT PYTORCH
        ''' 
        # Define the parameters for each loop_lstm call
        
        # lstm_params = [
        #     # (12, 12, 100, True, 16, 'direct'),
        #     (12, 12, 100, True, 16, 'recursive'),
        # ]
        
        # processes = []
        
        # # Start a new process for each set of parameters
        # for params in lstm_params:
        #     cmd = [
        #         "python", "run_lstm_script_pytorch.py",
        #         str(params[0]), str(params[1]), str(params[2]),
        #         str(params[3]), str(params[4]), str(params[5]),
        #     ]
        #     p = subprocess.Popen(cmd)
        #     processes.append(p)
        
        # # Wait for all processes to complete
        # for p in processes:
        #     p.wait()

        ''' 
        # INFO: TEST PYTORCH  =========== 5 ANOS ===========
        ''' 
        # from long_short_term_memory_pytorch_5_years import product_and_single_thread_testing
        # multiprocessing.set_start_method("spawn")
        # thread = multiprocessing.Process(target=product_and_single_thread_testing)
        # thread.start()
        # thread.join()

        ''' 
        # INFO: ALL STATES RECURSIVE/DIRECT PYTORCH =========== 5 ANOS ===========
        ''' 
        # Define the parameters for each loop_lstm call
        
        lstm_params = [
            (12, 12, 100, True, 16, 'direct'),
            # (12, 12, 100, True, 16, 'recursive'),            
        ]
        
        processes = []
        
        # Start a new process for each set of parameters
        for params in lstm_params:
            cmd = [
                "python", "run_lstm_script_pytorch_5_years.py",
                str(params[0]), str(params[1]), str(params[2]),
                str(params[3]), str(params[4]), str(params[5]),
            ]
            p = subprocess.Popen(cmd)
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.wait()
    
        lstm_params = [
            # (12, 12, 100, True, 16, 'direct'),
            (12, 12, 100, True, 16, 'recursive'),            
        ]
        
        processes = []
        
        # Start a new process for each set of parameters
        for params in lstm_params:
            cmd = [
                "python", "run_lstm_script_pytorch_5_years.py",
                str(params[0]), str(params[1]), str(params[2]),
                str(params[3]), str(params[4]), str(params[5]),
            ]
            p = subprocess.Popen(cmd)
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.wait()

    except Exception as e:
        print("An error occurred:", e)