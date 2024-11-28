import sys
from long_short_term_memory import run_lstm_in_thread

def main(forecast_steps, time_steps, epochs, verbose, bool_save, save_model, metric_monitor, batch_size, type_predictions, type_lstm):
    run_lstm_in_thread(forecast_steps=forecast_steps, time_steps=time_steps, epochs=epochs, verbose=verbose, bool_save=bool_save,  save_model=False, metric_monitor= metric_monitor, batch_size= batch_size, type_predictions=type_predictions, type_lstm=type_lstm)

if __name__ == "__main__":
    forecast_steps = int(sys.argv[1])
    time_steps = int(sys.argv[2])
    epochs = int(sys.argv[3])
    verbose = int(sys.argv[4])
    bool_save = sys.argv[5]
    save_model = sys.argv[6]
    metric_monitor = str(sys.argv[7].lower())
    batch_size =  int(sys.argv[8])
    type_predictions = str(sys.argv[9].lower())
    type_lstm = str(sys.argv[10].upper())

    main(forecast_steps, time_steps, epochs, verbose, bool_save, save_model, metric_monitor, batch_size, type_predictions, type_lstm)