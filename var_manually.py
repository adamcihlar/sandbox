
import numpy as np
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

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
       ones = np.ones((x.shape[0]))
       x = np.column_stack([ones, x])

    return x, y


def rmse(y_true, y_hat):
    return (np.mean((y_hat - y_true)**2))**(1/2)


def estimate_var(data, order=1, lr=0.0001, n_iter=20000, display_info=False):
    x, y = expand_data(data=data, order=order)

    beta = np.random.standard_normal((y.shape[1], x.shape[1]))
    for i in range(n_iter):
        grad = (beta.dot(x.T) - y.T).dot(x)
        beta -= (lr * grad)
        if (i % 1000 == 0):
            print(beta)
            print(rmse(y, x.dot(beta.T)))
    return beta.T


def impulse_response_func(beta, periods=20):
    x0 = np.ones(1)
    irf = np.zeros((beta.shape[1], periods))
    shock = np.ones(beta.shape[1])

    irf[:,0] = beta[0,:] + shock

    for i in range(1, irf.shape[1]):
        x = np.concatenate([x0, irf[:,i-1]])
        irf[:,i] = beta.T.dot(x.T)


if __name__=='__main__':

    data = generate_data()
    beta = estimate_var(data, order=1)

    model = VAR(data)
    model_fitted = model.fit(1)
    model_fitted.params
    irf = model_fitted.irf(10)

    irf.plot()
    plt.savefig('assets/irf.png')

    beta - model_fitted.params
