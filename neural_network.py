""" the neural network embeded in the DQN agent """

from tensorflow.python.keras import Sequential,Input, Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from config import Config


class NeuralNetwork:

    def __init__(self, input_ports=5,#11*Config().U+7,
                 output_ports=Config().n_actions,
                 num_neurons=(64, 32),     # 64 32 16
                 memory_step=8,
                 activation_function='relu'):

        self.input_ports = input_ports
        self.output_ports = output_ports
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.memory_step = memory_step

    def get_model(self, chooseNet):
        if chooseNet == 1:           # DQN
            model = Sequential()
            model.add(Dense(self.num_neurons[0], input_dim=self.input_ports, activation=self.activation_function))
            for j in range(1, len(self.num_neurons)):
                model.add(Dense(self.num_neurons[j], activation=self.activation_function))
            model.add(Dense(self.output_ports))

        else:                        # DRQN
            model = Sequential()
            model.add(Dense(self.num_neurons[0], input_shape=(self.memory_step, self.input_ports),
                            activation=self.activation_function))
            model.add(Dense(self.num_neurons[1], activation=self.activation_function))
            model.add(LSTM(self.num_neurons[2], activation=self.activation_function, unroll=1))
            model.add(Dense(self.output_ports))

        return model


