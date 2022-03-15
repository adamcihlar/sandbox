
import numpy as np

def generate_data(shape=(100, 2), with_bias=True):
    rand_mat = np.ones(shape)
    for i in range(shape[1]):
        rand_mat[:,i] *= np.random.randint(-10, 10)
        rand_mat[:,i] += [np.random.normal(0, 2) for _ in rand_mat[:,i]]

    return rand_mat


def expand_data(data, order=1, bias=True):
    shp = data.shape
    for i in range(data.shape[1]):
        for j in range(1, order+1):
            data = np.column_stack([
                data,
                np.pad(data[0:-j,i], (j, 0), 'constant',
                       constant_values=(np.nan))
            ])

    data = data[~np.isnan(data).any(axis=1)]
    y = data[:, 0:shp[1]]
    x = data[:, shp[1]:]

    if bias:
        ones = np.ones((x.shape[0], shp[1]))
        x = np.column_stack([ones, x])

    return x, y


def rmse(y_true, y_hat):
    return (np.mean((y_hat - y_true)**2))**(1/2)


def estimate_var(x, y, lr=0.0001, n_iter=1000):
    beta = np.random.standard_normal((y.shape[1], x.shape[1]))
    for i in range(n_iter):
        grad = (beta.dot(x.T) - y.T).dot(x)
        beta -= (lr * grad)
        if (i % 20 == 0):
            print(rmse(y, x.dot(beta.T)))
    return beta

if __name__=='__main__':

    data = generate_data()
    x, y = expand_data(data, order=1)

    beta = estimate_var(x, y)
