import numpy as np
from keras.utils import to_categorical


def transform_to_1d_index_vector(data_source, data_dest):
    assert np.min(data_source) >= 1
    for data in data_source:
        data_dest[data-1] = 1


def transform_to_index_vector(data_source, max_data):
    assert np.max(data_source) <= max_data
    assert np.min(data_source) >= 1
    data_shape = data_source.shape
    data_transform = None
    if len(data_shape) == 1:
        data_transform = np.zeros((max_data,))
        transform_to_1d_index_vector(data_source, data_transform)
    else:
        data_transform = np.zeros((len(data_source), max_data)) #no 0, index 0 means 1
        for row_index, row in enumerate(data_source):
            transform_to_1d_index_vector(row, data_transform[row_index])

    return data_transform

def increasing_segmentation_transform(data_source, segmentation_span=33, max_data=30, min_data=1, **kwargs):
    '''
    
    :param data_source: 
    :param segmentation_span: every sample length, len(x):segmentation_span-1, len(y):1
    :param kwargs: 
    :return: 
    '''
    data_len = len(data_source)
    assert segmentation_span > 0
    assert data_len >= segmentation_span
    data_index = 0
    segmentation_tail_index = lambda : data_index + segmentation_span - 1
    label_one_hot_encoding_func = kwargs.get('label_one_hot_encoding_func')
    if not label_one_hot_encoding_func:
        label_one_hot_encoding_func = lambda data:transform_to_index_vector(data, max_data)
    x_train = None
    y_train = None
    while segmentation_tail_index() < data_len:
        x_data = data_source[data_index : segmentation_tail_index()]
        y_data = data_source[segmentation_tail_index()]
        assert np.max(x_data) <= max_data
        assert np.min(x_data) >= min_data
        assert np.max(y_data) <= max_data
        assert np.min(y_data) >= min_data
        # process x_data(2d) to 1d data
        x_data = x_data.reshape(1, -1)
        if data_index == 0:
            x_train = x_data
            y_train = y_data
        else:
            x_train = np.vstack((x_train, x_data))
            y_train = np.vstack((y_train, y_data))

        data_index += 1
    # process y_data(1d) to 1d one hot vector data
    y_train = label_one_hot_encoding_func(y_train)
    return x_train, y_train



