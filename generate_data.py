import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from systems_collection import systems_collection, inits_collection

def generate_data(system, inits_setting='random', iinit=0, time_start=0, time_end=10, time_step=0.01,
                  snr=None, calculate_derivatives=False, path_out=".\\data\\", file_name="data.csv"):
    """
    Generate data by simulating dynamical systems

    Inputs:
        - system: object of class <System>
        - inits_setting: string; can be either 'random' (default) or 'collection':
                        'random': meaning that initial values are randomly chosen based on
                         the boundaries defined in the <System> object.
                        'collection': inits are taken from the dictionary "inits_collection".
        - time_start: float;
        - time_end: float;
        - time_step: float;
        - snr: float; desired Signal-to-Noise Ratio (in dB). If None, no noise will be added.
        - calculate_derivatives: bool; derivatives are added as additional columns in the csv file
        - path_out: string; path to save the data
        - file_name: string;
    Outputs:
        - data in form of pandas dataframe
        - data is saved in csv file
    """

    # set initial state
    if inits_setting == 'random':
        inits = system.get_inits()
    elif inits_setting == 'collection':
        inits = inits_collection[system.name][iinit]
    else:
        inits = system.get_inits()
        print('Inits_setting can be set either randomly (inits_setting="random") or defined in a dict (inits_setting="collection").'
              'No other options allowed. Default "random" is set.')

    # simulate
    ode_result = system.simulate(inits, time_start=time_start, time_end=time_end, time_step=time_step)
    simulation = ode_result.y.transpose()
    sim_times = np.reshape(ode_result.t, (-1, 1))

    # add noise
    noisy_data = add_noise(simulation, target_snr_db=snr)

    # calculate derivatives
    if calculate_derivatives:
        data = add_derivatives(noisy_data, time_step)
        column_names = ['t'] + system.state_vars + ['d' + i for i in system.state_vars]
    else:
        data = noisy_data
        column_names = ['t'] + system.state_vars

    # create dataframe
    data_df = pd.DataFrame(np.hstack((sim_times, data)), columns=column_names)

    # save results
    path_out_data = f"{path_out}{system.name}\\"
    os.makedirs(path_out_data, exist_ok=True)
    data_df.to_csv(path_out_data + file_name, index=False)

    return data_df


def add_noise(data, target_snr_db=None):
    # noise is set based on desired snr.

    if target_snr_db == None:
        return data

    powers = data ** 2  # Calculate signal power
    sig_avg_power = np.mean(powers, 0)  # Calculate signal avg power
    sig_avg_db = 10 * np.log10(sig_avg_power)  # Convert to dB
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_power = 10 ** (noise_avg_db / 10)

    n_vars = np.size(data, 1)
    data_length = len(data)
    data_out = np.empty_like(data)
    for ivar in range(n_vars):
        noise = np.random.normal(0, np.sqrt(noise_avg_power)[ivar], data_length)
        data_out[:, ivar] = data[:, ivar] + noise
    return data_out


def add_derivatives(data, time_step):
    X = np.copy(data)
    dX = np.array([np.gradient(Xi, time_step) for Xi in X.T]).transpose()
    return np.hstack((data, dX))

def plot_data(system, data, path_out=".\\data\\plots\\", file_name=f"figure.png", dpi=300):
    plt.figure()
    plt.subplot(1, 2, 1)
    [plt.plot(data.t, data[i]) for i in system.state_vars]
    plt.title('Signal time series')
    plt.ylabel('Voltage (V)')
    plt.xlabel('Time (s)')
    if len(system.state_vars) == 3:
        plt.subplot(1, 2, 2, projection='3d')
        plt.plot(data['x'], data['y'], data['z'])
    else:
        plt.subplot(1, 2, 2)
        plt.plot(data['x'], data['y'])
    plt.title('Phase plot')
    plt.ylabel('Y')
    plt.xlabel('X')

    path_out_fig = f"{path_out}{system.name}\\plots\\"
    os.makedirs(path_out_fig, exist_ok=True)
    plt.savefig(path_out_fig + file_name, dpi=dpi)
    plt.close('all')
    return 1


## MAIN
systems = systems_collection # systems that should be simulated
time_start = 0
time_end = 10
time_step = 0.1
times = np.arange(time_start, time_end, time_step)
snr = None
num_datasets = 4    # how many times it should be repeated for the same systems (different initial values)
calculate_derivatives = True
make_plots = False
path_out = ".\\data\\"

##

for system in systems:
    print(system)
    for iinit in range(num_datasets):
        file_name = f"{{}}_{systems[system].name}_len{time_end}_rate{str(time_step).replace('.', '')}_snr{snr}_init{iinit}.{{}}"
        data_name = file_name.format('data', 'csv')
        data = generate_data(systems[system], inits_setting='collection', iinit=iinit, time_start=time_start, time_end=time_end, time_step=time_step,
                             snr=snr, calculate_derivatives=calculate_derivatives, path_out=path_out, file_name=data_name)

        if make_plots:
            plot_name = file_name.format('figure', 'png')
            plot_data(systems[system], data, path_out=path_out, file_name=plot_name)


##
