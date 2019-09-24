# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import os, sys, platform, pickle
from os.path import expanduser
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

def get_glove_data():
    thisID = 'Sep21'
    if platform.system() == "Darwin":
        preproc_data_dir = f"/Users/ddchen/ThinkTank/Knowledge/PythonZens/ml2/{thisID}PreprocessDebug"
    else:
        preproc_data_dir = f"/home/ddchen/ml2/{thisID}PreprocessDebug"
    with open(f'{preproc_data_dir}/data.pkl', 'rb') as output:
        data = pickle.load(output)
        embedding = pickle.load(output)
    del data
    return embedding[0:400,:,:]

if __name__ == "__main__":
    # x = get_data()
    # x = x[0:10, :, 8:10]
    x = get_glove_data()
    input_dim = x.shape[-1] # 13
    timesteps = x.shape[1] # 3
    batch_size = 1

    vae, enc, gen = create_lstm_vae(input_dim,
        timesteps=timesteps,
        batch_size=batch_size,
        intermediate_dim=128,
        latent_dim=100,
        epsilon_std=1.)

    # vae.fit(x, x, batch_size=batch_size, epochs=6)
    vae.fit(x, batch_size=batch_size, epochs=10) #at least 2 epochs needed to make 8th in [:,2,8] work

#%%
def next_filepath(filepath_template):
    i = 1
    while os.path.exists(filepath_template.format(i)):
        i += 1
    return filepath_template.format(i)

#%%
preds = vae.predict(x, batch_size=batch_size)

timestep = 2
feature = 40
# pick a column to plot.
print("plotting...")
print("x: {}, preds: {}".format(x.shape, preds.shape))
# plt.plot(x[3,2,:], label='data')
# plt.plot(preds[3,2,:], label='predict')
plt.plot(x[0:100,timestep,feature], label='data')
plt.plot(preds[0:100,timestep,feature], label='predict')
plt.legend()
plt.show()
img_filepath = next_filepath(expanduser('~/Downloads/keras_lstm_vae/exampleGloVeEpoch10-{}.png'))
plt.savefig(img_filepath)
print(f"Saved {img_filepath}")


