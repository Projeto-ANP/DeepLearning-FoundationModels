from n_beats import run_nbeats_in_thread
from n_beats_5_years import run_nbeats_in_thread_5_years


if __name__ == "__main__":
    try:

        ''' 
        # INFO: ALL STATES N-BEATS 
        ''' 

        run_nbeats_in_thread(type_experiment=1)

        ''' 
        # INFO: ALL STATES N-BEATS 5 YEARS
        ''' 
        run_nbeats_in_thread_5_years(type_experiment=1)

    except Exception as e:
        print("An error occurred:", e)