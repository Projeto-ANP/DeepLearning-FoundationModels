import subprocess
import multiprocessing


if __name__ == "__main__":
    try:

        ''' 
        # INFO: TEST
        ''' 
        # from test import product_and_single_thread_testing
        # multiprocessing.set_start_method("spawn")
        # thread = multiprocessing.Process(target=product_and_single_thread_testing)
        # thread.start()
        # thread.join()

        ''' 
        # INFO: SINGLE STATE
        ''' 
        
        # from long_short_term_memory import product_and_single_thread_testing
        # multiprocessing.set_start_method("spawn")
        # thread = multiprocessing.Process(target=product_and_single_thread_testing)
        # thread.start()
        # thread.join()

        ''' 
        # INFO: ALL STATES RECURSIVE/DIRECT
        ''' 
        # # Define the parameters for each loop_lstm call
        # # forecast_steps, time_steps, epochs, verbose, bool_save, save_model, metric_monitor, batch_size, type_predictions, type_lstm
        
        # lstm_params = [
        #     (12, 12, 100, 1, True, False, 'loss', 16, 'recursive', 'LSTM-SIMPLES'),
        #     (12, 12, 100, 1, True, False, 'loss', 16, 'direct', 'LSTM-SIMPLES')
        # ]
        
        # processes = []
        
        # # Start a new process for each set of parameters
        # for params in lstm_params:
        #     cmd = [
        #         "python", "run_lstm_script.py",
        #         str(params[0]), str(params[1]), str(params[2]),
        #         str(params[3]), str(params[4]), str(params[5]), 
        #         str(params[6]), str(params[7]), str(params[8]),
        #         str(params[9]),
        #     ]
        #     p = subprocess.Popen(cmd)
        #     processes.append(p)
        
        # # Wait for all processes to complete
        # for p in processes:
        #     p.wait()

        ''' 
        # INFO: TEST FEATURES CATCH_22
        ''' 
        # from long_short_term_memory_features import product_and_single_thread_testing
        # multiprocessing.set_start_method("spawn")
        # thread = multiprocessing.Process(target=product_and_single_thread_testing)
        # thread.start()
        # thread.join()

        ''' 
        # INFO: ALL STATES RECURSIVE/DIRECT CATCH_22
        ''' 
        # # Define the parameters for each loop_lstm call
        # # forecast_steps, time_steps, epochs, verbose, bool_save, save_model, metric_monitor, batch_size, type_predictions, type_lstm
        
        # lstm_params = [
        #     (12, 12, 100, 1, True, False, 'loss', 16, 'recursive', 'LSTM-F'),
        #     (12, 12, 100, 1, True, False, 'loss', 16, 'direct_dense12', 'LSTM-F')
        # ]
        
        # processes = []
        
        # # Start a new process for each set of parameters
        # for params in lstm_params:
        #     cmd = [
        #         "python", "run_lstm_script.py",
        #         str(params[0]), str(params[1]), str(params[2]),
        #         str(params[3]), str(params[4]), str(params[5]), 
        #         str(params[6]), str(params[7]), str(params[8]),
        #         str(params[9]),
        #     ]
        #     p = subprocess.Popen(cmd)
        #     processes.append(p)
        
        # # Wait for all processes to complete
        # for p in processes:
        #     p.wait()

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
        # forecast_steps, time_steps, epochs, verbose, bool_save, save_model, batch_size, type_predictions, type_lstm
        
        # lstm_params = [
        #     (12, 12, 100, 1, True, False, 32, 'recursive', 'LSTM-PYTORCH'),
        #     (12, 12, 100, 1, True, False, 32, 'direct', 'LSTM-PYTORCH')
        # ]
        
        # processes = []
        
        # # Start a new process for each set of parameters
        # for params in lstm_params:
        #     cmd = [
        #         "python", "run_lstm_script_pytorch.py",
        #         str(params[0]), str(params[1]), str(params[2]),
        #         str(params[3]), str(params[4]), str(params[5]), 
        #         str(params[6]), str(params[7]), str(params[8]),
        #     ]
        #     p = subprocess.Popen(cmd)
        #     processes.append(p)
        
        # # Wait for all processes to complete
        # for p in processes:
        #     p.wait()

        ''' 
        # INFO: TEST PYTORCH  =========== 5 ANOS ===========
        ''' 
        # from long_short_term_memory_pytorch5ANOS import product_and_single_thread_testing
        # multiprocessing.set_start_method("spawn")
        # thread = multiprocessing.Process(target=product_and_single_thread_testing)
        # thread.start()
        # thread.join()

        ''' 
        # INFO: ALL STATES RECURSIVE/DIRECT PYTORCH =========== 5 ANOS ===========
        ''' 
        # Define the parameters for each loop_lstm call
        # forecast_steps, time_steps, epochs, verbose, bool_save, save_model, batch_size, type_predictions, type_lstm
        
        lstm_params = [
            (12, 12, 100, 1, True, False, 32, 'recursive', 'LSTM-PYTORCH'),
            (12, 12, 100, 1, True, False, 32, 'direct', 'LSTM-PYTORCH')
        ]
        
        processes = []
        
        # Start a new process for each set of parameters
        for params in lstm_params:
            cmd = [
                "python", "run_lstm_script_pytorch_5anos.py",
                str(params[0]), str(params[1]), str(params[2]),
                str(params[3]), str(params[4]), str(params[5]), 
                str(params[6]), str(params[7]), str(params[8]),
            ]
            p = subprocess.Popen(cmd)
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.wait()



    except Exception as e:
        print("An error occurred:", e)