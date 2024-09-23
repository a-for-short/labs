import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
import os

def main(filename):
    # Read and process the file
    file = open('data/'+filename)
    lines = file.readlines()
    file.close()

    # Data starts at line 10 (index 9)
    data_start = 0

    # Remove whitespaces and convert data to the correct type
    for i in range(data_start, len(lines) - 1):
        lines[i] = [el for el in lines[i].strip().replace(',', '.').split('\t') if el != '']
        for j in range(len(lines[i])):
            lines[i][j] = float(lines[i][j])

    # Now lines[i] = [entry number [1], time [ms], v_acc [0.01 mV], anode current [0.1 mA]]
    x_ind = 0
    y_ind = 1
    # Setup axes labels and units
    x_label = ''
    y_label = 'Интенсивность, у.е.' #it is in mA but as it doesnt really matter, i'll put a.u. there

    x_data = np.array([])
    y_data = np.array([])

    # Skip some entries at the beginning and end
    skip_first = 10
    skip_last = 10

    # Extract the data into arrays
    for i in range(data_start + skip_first, len(lines) - skip_last):
        x_data = np.append(x_data, lines[i][x_ind])
        y_data = np.append(y_data, lines[i][y_ind])

    # Sort data based on x_data
    data = np.array([x_data, y_data])
    ind_sort = np.argsort(data[0, :])
    data_sort = data[:, ind_sort]

    # Separate x and y data after sorting
    x_data_sort = data_sort[0, :]
    y_data_sort = data_sort[1, :]

    x_range = np.max(x_data_sort) - np.min(x_data_sort)
    y_range = np.max(y_data_sort) - np.min(y_data_sort)

    # Handle near-duplicate x values by averaging their corresponding y values
    tolerance = 1e-5  # Tolerance for considering x values as duplicates
    #Ensure correct datatype for rounding
    x_data_sort = np.array(x_data_sort, dtype=np.float64)
    y_data_sort = np.array(y_data_sort, dtype=np.float64)
    x_rounded = np.round(x_data_sort / tolerance) * tolerance

    x_unq, inverse_indices = np.unique(x_rounded, return_inverse=True)
    y_unq = np.bincount(inverse_indices, weights=y_data_sort) / np.bincount(inverse_indices)


    # Create a finer array of x values for plotting
    #nop = 500(
    #x_fine = np.linspace(np.min(x_unq), np.max(x_unq), num=nop)

    # Interpolation with strictly increasing x values
    #interp_func = interp1d(x_unq, y_unq, kind='cubic')
    #y_smooth = interp_func(x_fine)

    # Savitzky-Golay smoothing if needed
    #window_size = 35  # Should be an odd number
    #poly_order = 3
    #y_smooth = savgol_filter(y_smooth, window_size, poly_order)

    #Find minima bigger than 1.3
    top_indeces, _ = find_peaks(y_data_sort, height=y_range*0.13)  # Adjust 'distance' as needed

    # Plot the data
    plt.figure(figsize=(10, 6))

    #Make order of magnitude adjustments to the axes
    #x_fine = x_fine / 100
    #x_data_sort = x_data_sort / 100
    #y_smooth = y_smooth / 10
    #y_data_sort = y_data_sort / 10

    #Make pointers
    for maxi in top_indeces:
    # Calculate offsetы
        x_offset = x_range * 0.01  # Small offset for x
        y_offset = y_range * 0.1   # Larger offset for y

        plt.plot(x_data_sort[maxi], y_data_sort[maxi], "o", color='red')  # Plot the points
        plt.annotate(
            f'{x_data_sort[maxi]:.1f}',  # Annotate the point with the corresponding y value
            xy=(x_data_sort[maxi], y_data_sort[maxi]),  # Position the annotation at the point
            xytext=(x_data_sort[maxi] - x_offset, y_data_sort[maxi] + y_offset),  # Offset the annotation
            textcoords='data',
            arrowprops=dict(
                facecolor='black',
                arrowstyle='->',
                connectionstyle='arc3,rad=0'
            )
        )

    plt.plot(x_data_sort, y_data_sort, label='Original data', color='orange')
    #plt.plot(x_fine, y_smooth, label='Interpolated and smoothed data', color='blue')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend()
    name = filename
    plt.savefig(f'figures/{name}.png')
    plt.close()

for file in os.listdir('data'):
    main(file)