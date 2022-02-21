from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras import Model, regularizers


class Net:
    def __init__(self, state_size, action_size, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.optimizer = SGD(lr=learning_rate)

    def create_model(self):
        input = Input(shape=(self.state_size,))
        d1 = Dense(300, activation="relu", activity_regularizer=regularizers.l2())(input)
        dr1 = (Dropout(0.25))(d1)
        d2 = Dense(600, activation="relu", activity_regularizer=regularizers.l2())(dr1)
        dr2 = (Dropout(0.15))(d2)
        out = Dense(9, activation="Linear")(dr2)

        self.model = Model(inputs=input, outputs=out)
        self.model.compile(loss="mse", optimizer=self.optimizer, metrics=['Accuracy'])
        print(self.model.summary())
        return self.model
