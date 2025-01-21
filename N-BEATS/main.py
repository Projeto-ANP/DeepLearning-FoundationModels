import subprocess
import multiprocessing


if __name__ == "__main__":
    try:

        ''' 
        # INFO: ALL STATES N-BEATS 
        ''' 

        # all_params = [
        #     (12, 12, True),
        # ]
        
        # processes = []
        
        # # Start a new process for each set of parameters
        # for params in all_params:
        #     cmd = [
        #         "python", "run_all_n_beats_script.py",
        #         str(params[0]), str(params[1]), str(params[2]),
        #     ]
        #     p = subprocess.Popen(cmd)
        #     processes.append(p)
        
        # # Wait for all processes to complete
        # for p in processes:
        #     p.wait()

        ''' 
        # INFO: SINGLE STATE 5 YEARS
        ''' 
        # from n_beats import product_and_single_thread_testing
        # multiprocessing.set_start_method("spawn")
        # thread = multiprocessing.Process(target=product_and_single_thread_testing)
        # thread.start()
        # thread.join()


        ''' 
        # INFO: ALL STATES N-BEATS 5 YEARS
        ''' 

        all_params = [
            (12, 12, True),
        ]
        
        processes = []
        
        # Start a new process for each set of parameters
        for params in all_params:
            cmd = [
                "python", "run_all_n_beats_script_5_years.py",
                str(params[0]), str(params[1]), str(params[2]),
            ]
            p = subprocess.Popen(cmd)
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.wait()

    except Exception as e:
        print("An error occurred:", e)