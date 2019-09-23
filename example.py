import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)
from lstm_vae import create_lstm_vae

def get_data():
    # read data from file
    data = np.fromfile('sample_data.dat').reshape(419,13)
    timesteps = 3
    dataX = []
    for i in range(len(data) - timesteps - 1):
        x = data[i:(i+timesteps), :]
        dataX.append(x)
    return np.array(dataX)


if __name__ == "__main__":
    x = get_data()
    # x = x[0:10, :, 8:10]
    input_dim = x.shape[-1] # 13
    timesteps = x.shape[1] # 3
    batch_size = 1

    vae, enc, gen = create_lstm_vae(input_dim, 
        timesteps=timesteps, 
        batch_size=batch_size, 
        intermediate_dim=input_dim,
        latent_dim=100,
        epsilon_std=1.)

    # vae.fit(x, x, batch_size=batch_size, epochs=6)
    vae.fit(x, batch_size=batch_size, epochs=2) #at least 2 epochs needed to make 8th in [:,2,8] work

#%%
    preds = vae.predict(x, batch_size=batch_size)

    # pick a column to plot.
    print("plotting...")
    print("x: %s, preds: %s" % (x.shape, preds.shape))
    # plt.plot(x[3,2,:], label='data')
    # plt.plot(preds[3,2,:], label='predict')
    plt.plot(x[:,2,8], label='data')
    plt.plot(preds[:,2,8], label='predict')
    plt.legend()
    plt.show()


