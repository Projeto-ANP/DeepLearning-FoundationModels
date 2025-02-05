import sys
from model_time_moe_5_years import run_all_in_thread_5_years

def main(type_model):
    run_all_in_thread_5_years(type_model=type_model)

if __name__ == "__main__":
    type_model = str(sys.argv[1])

    main(type_model)