import subprocess
import multiprocessing


if __name__ == "__main__":
    try:

        ''' 
        # INFO: SINGLE STATE
        ''' 
        
        # from model_time_moe import product_and_single_thread_testing
        # multiprocessing.set_start_method("spawn")
        # thread = multiprocessing.Process(target=product_and_single_thread_testing)
        # thread.start()
        # thread.join()

        ''' 
        # INFO: ALL STATES TimeMoE-50M/TimeMoE-200M
        ''' 

        # forecast_steps=forecast_steps, time_steps=time_steps, bool_save=bool_save, type_model=type_model
        
       
        time_moe_params = [
            # (12, 12, True, 'TimeMoE-50M-FINE-TUNING'),
            (12, 12, True, 'TimeMoE-200M-FINE-TUNING'),
        ]
        
        processes = []
        
        # Start a new process for each set of parameters
        for params in time_moe_params:
            cmd = [
                "python", "run_all_time_moe_script.py",
                str(params[0]), str(params[1]), str(params[2]),
                str(params[3]),
            ]
            p = subprocess.Popen(cmd)
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.wait()
        
        ''' 
        # INFO: SINGLE STATE 5 YEARS
        ''' 
        # from model_time_moe_5_years import product_and_single_thread_testing_5_years
        # multiprocessing.set_start_method("spawn")
        # thread = multiprocessing.Process(target=product_and_single_thread_testing_5_years)
        # thread.start()
        # thread.join()


        ''' 
        # INFO: ALL STATES TimeMoE-50M/TimeMoE-200M 5 YEARS
        ''' 

        # forecast_steps=forecast_steps, time_steps=time_steps, bool_save=bool_save, type_model=type_model

        # time_moe_params = [
        #     (12, 12, True, 'TimeMoE-50M_ZERO_SHOT'),
        #     (12, 12, True, 'TimeMoE-200M_ZERO_SHOT'),
        # ]
        
        # processes = []
        
        # # Start a new process for each set of parameters
        # for params in time_moe_params:
        #     cmd = [
        #         "python", "run_all_time_moe_script_5_years.py",
        #         str(params[0]), str(params[1]), str(params[2]),
        #         str(params[3]),
        #     ]
        #     p = subprocess.Popen(cmd)
        #     processes.append(p)
        
        # # Wait for all processes to complete
        # for p in processes:
        #     p.wait()

    except Exception as e:
        print("An error occurred:", e)