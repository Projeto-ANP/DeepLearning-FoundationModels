import sys
from model_time_moe_5_years import run_all_in_thread_5_years

def main(forecast_steps, time_steps, bool_save, type_model):
    run_all_in_thread_5_years(forecast_steps=forecast_steps, time_steps=time_steps, bool_save=bool_save, type_model=type_model)

if __name__ == "__main__":
    forecast_steps = int(sys.argv[1])
    time_steps = int(sys.argv[2])
    bool_save = str(sys.argv[3])
    type_model = str(sys.argv[4])

    main(forecast_steps, time_steps, bool_save, type_model)