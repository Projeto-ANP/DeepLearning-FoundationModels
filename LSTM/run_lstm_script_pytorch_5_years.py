import sys
from long_short_term_memory_pytorch_5_years import run_lstm_in_thread

def main(forecast_steps, time_steps, epochs, bool_save, batch_size, type_predictions):
    run_lstm_in_thread(forecast_steps=forecast_steps, time_steps=time_steps, epochs=epochs, bool_save=bool_save, batch_size= batch_size, type_predictions=type_predictions)

if __name__ == "__main__":
    forecast_steps = int(sys.argv[1])
    time_steps = int(sys.argv[2])
    epochs = int(sys.argv[3])
    bool_save = sys.argv[4]
    batch_size =  int(sys.argv[5])
    type_predictions = str(sys.argv[6].lower())

    main(forecast_steps, time_steps, epochs, bool_save, batch_size, type_predictions)