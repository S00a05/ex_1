import os
import json
#from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
def main():
    print("Hello World.")
if __name__ == "__main__":
    main()

# Task 1 ==============================================================================
def read_parameters(params_file_path: str) -> dict:
    
    try:
        with open(params_file_path, 'r') as file:
            params_data = json.load(file)
        return params_data
    except FileNotFoundError:
        print(f"File not found: {params_file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return {}

params_file_path = "D:\Coding\Ex1\ex_1\code_modules\python\Ex1\params.json"  
params_data = read_parameters(params_file_path)

print(params_data)


# Task 2 ============================================================================


def calculate_resonance_frequency(L, C):
    return 1 / (2 * np.pi * np.sqrt(L * C))

def calculate_magnitude(frequency, R, L, C):
    omega = 2 * np.pi * frequency
    impedance = np.sqrt((R ** 2) + (omega * L - 1 / (omega * C)) ** 2)
    magnitude = 20 * np.log10(1 / impedance)
    return magnitude

def calculate_phase(frequency, R, L, C):
    omega = 2 * np.pi * frequency
    phase = np.arctan((omega * L - 1 / (omega * C)) / R)
    phase = np.degrees(phase)
    return phase

def plot_bode(R, L, C, destination_dir="."):
    resonance_frequency = calculate_resonance_frequency(L, C)
    frequency_range = np.logspace(np.log10(resonance_frequency) - 1, np.log10(resonance_frequency) + 1, num=1000)
    
    magnitude_at_resonance = calculate_magnitude(resonance_frequency, R, L, C)
    phase_at_resonance = calculate_phase(resonance_frequency, R, L, C)

    magnitude = calculate_magnitude(frequency_range, R, L, C)
    phase = calculate_phase(frequency_range, R, L, C)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    
    ax1.semilogx(frequency_range, magnitude, label='Magnitude', color='blue')
    ax1.scatter(resonance_frequency, magnitude_at_resonance, color='red', marker='o')
    ax1.axvline(resonance_frequency, color='orange', linestyle='--')

    ax1.set_title('Magnitude Plot')
    ax1.set_xlabel('Frequency in Hz')
    ax1.set_ylabel('|Y(jω)| in Ω')
    ax1.grid(True)

    
    ax2.semilogx(frequency_range, phase, label='Phase', color='green')
    ax2.scatter(resonance_frequency, phase_at_resonance, color='red', marker='o')
    ax2.axvline(resonance_frequency, color='orange', linestyle='--')
    
    ax2.set_title('Phase Plot')
    ax2.set_xlabel('Frequency in Hz')
    ax2.set_ylabel('{Y(jω)} in °')
    ax2.grid(True)
    
    plt.subplots_adjust(hspace=0.5)  
    fig.suptitle('Bode Plot for Series Resonant Circuit')

    plot_filename = os.path.join(destination_dir, 'Ex1_Team05_series_resonant_bode.png')
    plt.savefig(plot_filename)

    return fig

params_file_path = "D:\Coding\Ex1\ex_1\code_modules\python\Ex1\params.json"
parameters = read_parameters(params_file_path)
R = parameters['R']
L = parameters['L']
C = parameters['C']
destination_dir = r"D:\Coding\Ex1\ex_1\code_modules\python\Ex1\Results"  

figure = plot_bode(R, L, C, destination_dir)
plt.show()

# Task 3 =================================================================

def perform_monte_carlo_simulation(R, L, C, sigma_R, sigma_L, sigma_C, N_samples, destination_dir="."):
    rng_L = np.random.default_rng(11223344)
    rng_C = np.random.default_rng(22334455)
    rng_R = np.random.default_rng(33445566)

    L_samples = rng_L.normal(L, sigma_L, N_samples)
    C_samples = rng_C.normal(C, sigma_C, N_samples)
    R_samples = rng_R.uniform(R - np.sqrt(3) * sigma_R, R + np.sqrt(3) * sigma_R, N_samples)

    fig, ax = plt.subplots(3, 1, figsize=(8, 10))
    ax[0].hist(L_samples, bins=50, color='blue', alpha=0.7)
    ax[0].set_title('Histogram of L')
    ax[0].set_xlabel('L (Henry)')
    ax[0].set_ylabel('Frequency')
    ax[1].hist(C_samples, bins=50, color='green', alpha=0.7)
    ax[1].set_title('Histogram of C')
    ax[1].set_xlabel('C (Farad)')
    ax[1].set_ylabel('Frequency')
    ax[2].hist(R_samples, bins=50, color='red', alpha=0.7)
    ax[2].set_title('Histogram of R')
    ax[2].set_xlabel('R (Ohm)')
    ax[2].set_ylabel('Frequency')

    plt.tight_layout()

    mean_L = np.mean(L_samples)
    std_dev_L = np.std(L_samples)
    mean_C = np.mean(C_samples)
    std_dev_C = np.std(C_samples)
    min_R = np.min(R_samples)
    max_R = np.max(R_samples)

    f0_samples = 1 / (2 * np.pi * np.sqrt(L_samples * C_samples))
    Q_samples = np.sqrt(L_samples / C_samples) / R_samples

    fig2, ax2 = plt.subplots(2, 1, figsize=(8, 6))
    ax2[0].hist(f0_samples, bins=50, color='blue', alpha=0.7)
    ax2[0].set_title('Histogram of f0')
    ax2[0].set_xlabel('f0 (Hz)')
    ax2[0].set_ylabel('Frequency')
    ax2[1].hist(Q_samples, bins=50, color='green', alpha=0.7)
    ax2[1].set_title('Histogram of Q')
    ax2[1].set_xlabel('Q')
    ax2[1].set_ylabel('Frequency')

    plt.tight_layout()

    mean_f0 = np.mean(f0_samples)
    std_dev_f0 = np.std(f0_samples)
    mean_Q = np.mean(Q_samples)
    std_dev_Q = np.std(Q_samples)

    hist_filename_L = os.path.join(destination_dir, 'Ex1_Team05_series_resonant_RV_L_hist.png')
    hist_filename_C = os.path.join(destination_dir, 'Ex1_Team05_series_resonant_RV_C_hist.png')
    hist_filename_R = os.path.join(destination_dir, 'Ex1_Team05_series_resonant_RV_R_hist.png')
    hist_filename_f0 = os.path.join(destination_dir, 'Ex1_Team05_series_resonant_RV_f0_hist.png')
    hist_filename_Q = os.path.join(destination_dir, 'Ex1_Team05_series_resonant_RV_Q_hist.png')

    fig.savefig(hist_filename_L)
    fig2.savefig(hist_filename_C)
    fig.savefig(hist_filename_R)
    fig2.savefig(hist_filename_f0)
    fig2.savefig(hist_filename_Q)

    results_filename = os.path.join(destination_dir, 'Ex1_Team05_monte_carlo_results.json')
    results_data = {
        "f0_mean": mean_f0,
        "sigma_f0": std_dev_f0,
        "Q_mean": mean_Q,
        "sigma_Q": std_dev_Q
    }

    with open(results_filename, 'w') as results_file:
        json.dump(results_data, results_file, indent=4)

    return fig, fig2


params_file_path = "D:\Coding\Ex1\ex_1\code_modules\python\Ex1\params.json"
parameters = read_parameters(params_file_path)
R = parameters['R']
L = parameters['L']
C = parameters['C']
sigma_R = parameters['sigma_R']
sigma_L = parameters['sigma_L']
sigma_C = parameters['sigma_C']
destination_dir = "D:\Coding\Ex1\ex_1\code_modules\python\Ex1\Results"  # Use 'r' before the path to handle backslashes
N_samples = 10000

figure, figure2 = perform_monte_carlo_simulation(R, L, C, sigma_R, sigma_L, sigma_C, N_samples, destination_dir)
plt.show()
