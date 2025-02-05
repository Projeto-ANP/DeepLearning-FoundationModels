import sys
from n_beats_5_years5 import run_nbeats_in_thread

def main(forecast_steps, time_steps, bool_save):
    run_nbeats_in_thread(forecast_steps=forecast_steps, time_steps=time_steps, bool_save=bool_save)

if __name__ == "__main__":
    forecast_steps = int(sys.argv[1])
    time_steps = int(sys.argv[2])
    bool_save = str(sys.argv[3])

    main(forecast_steps, time_steps, bool_save)