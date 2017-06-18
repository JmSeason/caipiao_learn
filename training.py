import train_model
import data_transform
import data_source

def train_7lc():
    ds = data_source.get_7lc_winning_numbers()
    x_train, y_train = data_transform.increasing_segmentation_transform(ds,
                                                                        segmentation_span=33,
                                                                        max_data=30)
    model = train_model.mlp(x_train, y_train, hidden_layer_nodes=(100, 100),
                            output_layer_activation='sigmoid')
    model.train()



train_7lc()