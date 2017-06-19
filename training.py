import train_model
import data_transform
import data_source

def train_7lc():
    ds = data_source.get_7lc_winning_numbers()
    x_train, y_train = data_transform.increasing_segmentation_transform(ds,
                                                                        segmentation_span=20,
                                                                        max_data=30)
    model = train_model.mlp(x_train, y_train, hidden_layer_nodes=(2000, 300),
                            output_layer_activation='sigmoid')
    model.train(epochs=10, batch_size=2)



train_7lc()