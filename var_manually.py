
import numpy as np

def generate_data(shape=(100, 2), with_bias=True):
    rand_mat = np.ones(shape)
    for i in range(shape[1]):
        rand_mat[:,i] *= np.random.randint(-10, 10)
        rand_mat[:,i] += [np.random.normal(0, 2) for _ in rand_mat[:,i]]

    return rand_mat


def expand_data(data, order=1, bias=True):
    for i in range(data.shape[1]):
        for j in range(1, order+1):
            data = np.column_stack([
                data,
                np.pad(data[0:-j,i], (j, 0), 'constant',
                       constant_values=(np.nan))
            ])

#     if with_bias:
#         ones = np.ones(shape[0])
#         data = np.column_stack([ones, ones, data])

    return data

if __name__=='__main__':

    data = generate_data()
    data = expand_data(data, order=2)
