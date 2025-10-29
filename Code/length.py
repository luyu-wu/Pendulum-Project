import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sgn
import smplotlib  # noqa
from numpy import genfromtxt
from scipy.interpolate import interp1d
from numpy import linalg as la

for i in range(1, 7):
    my_data = genfromtxt("l" + str(i) + ".csv", delimiter=",")
    my_data[:, 0] -= my_data[0, 0]  # Set time baseline

    length = np.average(la.norm(my_data[:, 1:3], axis=1))
    # print("Length:", length)
    # print("Length Err:", np.std(la.norm(my_data[:, 1:3], axis=1)))
    b_error = 0.0075 / (length * 2)
    time = my_data[:, 0]
    x_displacement = my_data[:, 1]
    x_displacement -= np.average(x_displacement)
    y_displacement = my_data[:, 2]
    x_displacement_filtered = sgn.savgol_filter(x_displacement, 31, 3)
    y_displacement_filtered = sgn.savgol_filter(y_displacement, 31, 3)
    rate = 1000  # points per second
    # Interpolate data
    f_x, f_y = (
        interp1d(
            time,
            x_displacement_filtered,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        ),
        interp1d(
            time,
            y_displacement_filtered,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        ),
    )
    time = np.linspace(0, time.max(), int(time.max() * rate))
    dt = 1 / rate  # time step
    x_displacement, y_displacement = f_x(time), f_y(time)

    theta = np.arctan(x_displacement / y_displacement)
    theta -= np.average(theta)

    peaks, _ = sgn.find_peaks(theta)
    peaks_b, _ = sgn.find_peaks(-theta)
    if len(peaks) < len(peaks_b):
        peaks_b = peaks_b[: len(peaks)]
    else:
        peaks = peaks[: len(peaks_b)]
    peak_times, peak_times_b = time[peaks], time[peaks_b]
    peak_amplitudes, peak_amplitudes_b = np.abs(theta[peaks]), np.abs(theta[peaks_b])
    periods = np.concatenate((np.diff(time[peaks]), np.diff(time[peaks_b])))

    peak_times = np.vstack((peak_times, peak_times_b)).ravel("F")
    peak_amplitudes = np.vstack((peak_amplitudes, peak_amplitudes_b)).ravel("F")

    """
    first, last = (
        np.where(peak_amplitudes > 0.1)[0][-1],
        np.where(peak_amplitudes > 0.0455)[0][-1],
    )
    peak_amplitudes, peak_times = peak_amplitudes[first:last], peak_times[first:last]
    peak_times -= peak_times[0]
    """
    print(
        0.5
        * len(peak_amplitudes)
        * np.pi
        / np.log(np.average(peak_amplitudes[-3:-1]) / np.average(peak_amplitudes[0:2]))
    )

    decay_fit, cov = np.polyfit(peak_times, np.log(peak_amplitudes), 1, cov=True)
    fit_err = np.sqrt(np.diag(cov))

    omega_0 = 2 * np.pi / np.average(periods)
    Q_factor = omega_0 / (-2 * decay_fit[0])
    Q_fac_error = abs(Q_factor - (omega_0 / (-2 * (decay_fit[0] - fit_err[0]))))
    # print("Q-fac:", Q_factor)

    # print("Q-fac Error:", Q_fac_error)
    t_fit = np.linspace(-10, np.max(peak_times) + 10, 20)
    amplitude_fit = np.exp(np.polyval(decay_fit, t_fit))

    # print("Period:", np.average(periods))
    # print("Error:", np.std(periods) / np.sqrt(len(periods)))

    plt.plot(
        peak_times,
        peak_amplitudes,
        "o",
        # yerr=b_error,
        markersize=4,
        alpha=0.1,
        c="black",
        # label=str(i),
    )

    plt.plot(
        t_fit,
        amplitude_fit,
        "--",
        label=str(i)
        + " Exponential Fit ($Q = "
        + str(int(Q_factor))
        + "\\pm"
        + str(int(Q_fac_error * 10) / 10)
        + "$)",
    )

plt.xlabel("Time (s)")
plt.ylabel("Amplitude ($rad$)")
# plt.ylim(np.min(peak_amplitudes) * 0.95, np.max(theta) * 1.05)
# plt.xlim(-10, np.max(peak_times) + 10)
plt.yscale("log")
plt.grid(alpha=0.5)
plt.legend()
plt.show()
