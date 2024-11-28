import tensorflow as tf
from keras import Sequential, Input, Model
from keras.layers import Layer, LSTM, Dropout, Dense, Bidirectional, BatchNormalization, Attention, Concatenate, LayerNormalization, Flatten, TimeDistributed, ConvLSTM2D, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D # type: ignore
from kerastuner import HyperModel
import gc
import keras.backend as K  # type: ignore

class ModelBuilder(HyperModel):
    def __init__(self, time_steps, dense_predictions, type_lstm):
        """
        Initializes the ModelBuilder object.

        Args:
        - time_steps (int): Size of the input time_steps for the model.
        - dense_predictions (int): number of predictions to be made.
        """
        self.time_steps = time_steps
        self.dense_predictions = dense_predictions
        self.type_lstm = type_lstm

    def build(self, hp):
        """
        Builds a deep learning model based on hyperparameter choices.

        Args:
        - hp: Hyperparameters object from Kerastuner.

        Returns:
        - model: Compiled Keras model.
        """

        gc.collect()
        K.clear_session()
        tf.compat.v1.reset_default_graph()

        # Define hyperparameters
        params = {
            'time_steps': self.time_steps, 
            # 'num1_lstm': hp.Choice('num1_lstm', values= list(range(12, 129, 2))), 
            # 'num2_lstm': hp.Choice('num2_lstm', values= list(range(12, 129, 2))), 
        }

        model = self.create_model(params)

        return model
    
    def create_model(self, params):
        """
        Creates a Keras sequential model based on given parameters.

        Args:
        - params (dict): Dictionary containing model configuration parameters.

        Returns:
        - model: Compiled Keras model.
        """

        if self.type_lstm == "LSTM":
            strategy = tf.distribute.get_strategy()
            # strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                tf.keras.backend.clear_session()
                gc.collect()

                model = Sequential([
                    LSTM(128, return_sequences=True, input_shape=(self.time_steps, 1)),
                    LSTM(128, return_sequences=False),
                    Dense(self.dense_predictions)
                ])

                # Compile the model with optimizer, loss, and metrics
                opt = tf.keras.optimizers.Adam(learning_rate=0.01)
                model.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.MeanAbsolutePercentageError(), tf.keras.metrics.RootMeanSquaredError()])

                return model
            
        elif self.type_lstm == "LSTM-F":
            strategy = tf.distribute.get_strategy()
            # strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                tf.keras.backend.clear_session()
                gc.collect()

                model = Sequential([
                    LSTM(params['num1_lstm'], return_sequences=True, input_shape=(self.time_steps, 1)),
                    LSTM(params['num2_lstm'], return_sequences=False, input_shape=(params['num1_lstm'], 1)),
                    Dense(self.dense_predictions)
                ])

                # Compile the model with optimizer, loss, and metrics
                opt = tf.keras.optimizers.Adam(learning_rate=0.01)
                model.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.MeanAbsolutePercentageError(), tf.keras.metrics.RootMeanSquaredError()])

                return model
                        
        elif self.type_lstm == "LSTM-SIMPLES":
            strategy = tf.distribute.get_strategy()
            # strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                tf.keras.backend.clear_session()
                gc.collect()

                model = Sequential([
                    LSTM(params['num1_lstm'], activation='relu', return_sequences=False, input_shape=(self.time_steps, 1)),
                    Dense(self.dense_predictions)
                ])

                # Compile the model with optimizer, loss, and metrics
                opt = tf.keras.optimizers.Adam(learning_rate=0.01)
                model.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.MeanAbsolutePercentageError(), tf.keras.metrics.RootMeanSquaredError()])

                return model
        
        elif self.type_lstm == "BI-LSTM":
            strategy = tf.distribute.get_strategy()
            # strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                tf.keras.backend.clear_session()
                gc.collect()

                model = Sequential([
                    Bidirectional(LSTM(params['num1_lstm'], activation=params['activation'], return_sequences=True), input_shape=(self.time_steps, 1)),
                    Dropout(params['val_dropout']),
                    Bidirectional(LSTM(params['num2_lstm'], activation=params['activation'])),
                    Dropout(params['val_dropout']),
                    Dense(self.dense_predictions, activation=params['activation_dense'])
                ])

                # Compile the model with optimizer, loss, and metrics
                opt = tf.keras.optimizers.Adam(learning_rate=0.01)
                model.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.MeanAbsolutePercentageError(), tf.keras.metrics.RootMeanSquaredError()])

                return model
        
        elif self.type_lstm == "BI-LSTM2":
            strategy = tf.distribute.get_strategy()
            # strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                tf.keras.backend.clear_session()
                gc.collect()

                model = Sequential([
                    # Bidirectional(LSTM(units=36, input_shape=(self.time_steps, 1), return_sequences=True)),
                    # Bidirectional(LSTM(units=12, return_sequences=False)),
                    # Dense(1)
                    Bidirectional(LSTM(units=36, input_shape=(self.time_steps, 1), return_sequences=True)),
                    Bidirectional(LSTM(units=12, return_sequences=False)),
                    Dense(1)
                ])

                # Compile the model with optimizer, loss, and metrics
                opt = tf.keras.optimizers.Adam(learning_rate=0.01)
                model.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.MeanAbsolutePercentageError(), tf.keras.metrics.RootMeanSquaredError()])

                return model
        
        elif self.type_lstm == "LSTM-AM":
            strategy = tf.distribute.get_strategy()
            # strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                tf.keras.backend.clear_session()
                gc.collect()

                # Definindo o modelo com a camada de Attention do Keras
                inputs = Input(shape=(self.time_steps, 1))  # Defina o tamanho do input (ex: time_steps x features)

                # Primeira camada LSTM
                x = LSTM(units=36, return_sequences=True)(inputs)

                # Atenção
                attention = Attention()([x, x])  # Aplica atenção às saídas de LSTM (query, value)

                # Segunda camada LSTM 
                x = LSTM(units=12, return_sequences=False)(attention)

                # Saída final
                output = Dense(1)(x)

                # Construindo o modelo
                model = tf.keras.models.Model(inputs=inputs, outputs=output)

                # Compile the model with optimizer, loss, and metrics
                opt = tf.keras.optimizers.Adam(learning_rate=0.01)
                model.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.MeanAbsolutePercentageError(), tf.keras.metrics.RootMeanSquaredError()])

                return model