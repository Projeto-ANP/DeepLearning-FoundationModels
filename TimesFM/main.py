###  Code Description

'''

NOTE: Uncomment and comment out the code below as needed.

'''

from times_fm_code_forecasting import product_and_single_thread_testing, run_all_in_thread

if __name__ == "__main__":
    try:
        

        ''' 
        # INFO: SINGLE STATE (Testing model for a single state)
        ''' 
        # product_and_single_thread_testing()

        ''' 
        # INFO: ALL STATES (Testing model for all states)
        
        model: 200M or 500M
        
        ''' 
        run_all_in_thread(type_model='500M')

    except Exception as e:
        print("An error occurred:", e)