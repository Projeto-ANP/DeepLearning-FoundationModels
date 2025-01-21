###  Code Description

'''

The code is responsible for training the **Time-MoE** models with the ANP data and saving the results locally. Depending on the selected option, the code performs different types of forecasting tests. The test options include:

1. **SINGLE STATE**: This test forecasts the next 12 months for a single state and a single product. The forecast is made for one year ahead.
2. **ALL STATES**: The model is trained and makes predictions for the next 12 months (1 year), but for all the states and products of the ANP, performing a more comprehensive forecast.
3. **SINGLE STATE 5 YEARS**: This option makes predictions for 5 years ahead (i.e., 60 points), using the Time-MoE model to forecast future values over a longer period, for a single state and product.
4. **ALL STATES TimeMoE-50M / TimeMoE-200M 5 YEARS**: This setting tests the Time-MoE model with different scales (50M and 200M) to predict 60 points ahead, using all states and products.

These options allow testing the model in different forecasting scenarios, with the ability to generate predictions both for the short and long term, and can be adapted depending on the analysis needs.

NOTE: Uncomment and comment out the code below as needed.

'''

import subprocess
import multiprocessing
from model_time_moe_5_years import product_and_single_thread_testing_5_years
from model_time_moe import product_and_single_thread_testing


if __name__ == "__main__":
    try:

        ''' 
        # INFO: SINGLE STATE (Testing model for a single state)
        ''' 
        
        # multiprocessing.set_start_method("spawn")
        # thread = multiprocessing.Process(target=product_and_single_thread_testing)
        # thread.start()
        # thread.join()

        ''' 
        # INFO: ALL STATES TimeMoE-50M/TimeMoE-200M (Testing model for all states)
        # INFO: (input variables) forecast_steps, time_steps, bool_save, type_model 
        ''' 
        
        # time_moe_params = [
        #     (12, 12, True, 'TimeMoE-50M-FINE-TUNING'),
        #     (12, 12, True, 'TimeMoE-200M-FINE-TUNING'),
        # ]
        
        # processes = []
        
        # for params in time_moe_params:
        #     cmd = [
        #         "python", "run_all_time_moe_script.py",
        #         str(params[0]), str(params[1]), str(params[2]),
        #         str(params[3]),
        #     ]
        #     p = subprocess.Popen(cmd)
        #     processes.append(p)
        
        # for p in processes:
        #     p.wait()
        
        ''' 
        # INFO: SINGLE STATE 5 YEARS (Testing model for a single state over 5 years)
        ''' 
        
        # multiprocessing.set_start_method("spawn")
        # thread = multiprocessing.Process(target=product_and_single_thread_testing_5_years)
        # thread.start()
        # thread.join()


        ''' 
        # INFO: ALL STATES TimeMoE-50M/TimeMoE-200M 5 YEARS (Testing model for all states over 5 years)
        # INFO: (input variables) forecast_steps, time_steps, bool_save, type_model 
        ''' 

        time_moe_params = [
            (12, 12, True, 'TimeMoE-50M_FINE_TUNING_GLOBAL'),
            (12, 12, True, 'TimeMoE-200M_FINE_TUNING_INDIV'),
        ]
        
        processes = []
        
        for params in time_moe_params:
            cmd = [
                "python", "run_all_time_moe_script_5_years.py",
                str(params[0]), str(params[1]), str(params[2]),
                str(params[3]),
            ]
            p = subprocess.Popen(cmd)
            processes.append(p)
        
        for p in processes:
            p.wait()

    except Exception as e:
        print("An error occurred:", e)