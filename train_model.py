from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

class train_model(object):
    def __init__(self, input_data, y_labels):
        self.input_data = input_data
        self.y_labels = y_labels
        self.input_dim = input_data.shape[-1]
        self.output_dim = y_labels.shape[-1]
        self.model = None

    def train(self):
        pass

class mlp(train_model):
    def __init__(self, input_data, y_labels, **kwarg):
        '''
        
        :param input_data: 
        :param y_labels: 
        :param kwarg: {'hidden_layer_nodes': tuple, }
        '''
        super(mlp, self).__init__(input_data, y_labels)
        model = Sequential()
        dense_layers = kwarg.get('hidden_layer_nodes')
        default_activation = kwarg.get('default_activation')
        if not dense_layers:
            dense_layers = (10,)
        if not default_activation:
            default_activation = 'relu'
        for index,layer_node_num in enumerate(dense_layers):
            activation = kwarg.get('layer'+str(index)+'_activation')
            if not activation:
                activation = default_activation
            if index == 0:
                model.add(Dense(layer_node_num, activation=activation, input_dim=self.input_dim))
            else:
                model.add(Dense(layer_node_num, activation=activation))
        activation = kwarg.get('output_layer_activation')
        if not activation:
            activation = default_activation
        model.add(Dense(self.output_dim, activation=activation))
        self.model = model

    def train(self, **kwargs):
        optimizer = kwargs.get('optimizer')
        if not optimizer:
            optimizer = Adam(lr=1e-4)
        loss = kwargs.get('loss')
        if not loss:
            loss = 'categorical_crossentropy'
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=['accuracy'])
        epochs = kwargs.get('epochs')
        if not epochs:
            epochs = 10
        batch_size = kwargs.get('batch_size')
        if not batch_size:
            batch_size = 8
        self.model.fit(self.input_data,
                       self.y_labels,
                       epochs=epochs,
                       batch_size=batch_size)
