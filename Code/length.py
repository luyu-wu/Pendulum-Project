import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sgn
import smplotlib  # noqa
from numpy import genfromtxt
from scipy.interpolate import interp1d
from numpy import linalg as la

my_data = genfromtxt("l5.csv", delimiter=",")
my_data[:, 0] -= my_data[0, 0]  # Set time baseline

length = np.average(la.norm(my_data[:,1:3],axis=1))  # length in m for trig stuff lol
print("Length:",length)
print("Length Err:",np.std(la.norm(my_data[:,1:3],axis=1)))
b_error = 0.0075 / (length * 2)
# angle and energy
fig, ax1 = plt.subplots()
time = my_data[:, 0]
x_displacement = my_data[:, 1]
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

theta = np.arctan(x_displacement / y_displacement)  # np.arcsin(x_displacement / length)

peaks, _ = sgn.find_peaks(theta)
peaks_b, _ = sgn.find_peaks(-theta)

ax1.plot(time, theta)
ax1.plot(time[peaks], theta[peaks], "ro", markersize=4, label="Peaks")
ax1.plot(time[peaks_b], theta[peaks_b], "ro", markersize=4)
ax1.grid(alpha=0.5)

ax1.set_ylabel("Theta ($\\Theta$)")
ax1.set_xlabel("Time (s)")
ax1.legend()
ax1.set_title("Pendulum Angle vs. Time")
plt.tight_layout()
plt.show()



peak_times = time[peaks]
peak_amplitudes = np.abs(theta[peaks])
decay_fit, cov = np.polyfit(peak_times, np.log(peak_amplitudes), 1, cov=True)
fit_err = np.sqrt(np.diag(cov))

omega_0 = 2 * np.pi * (1 / (np.average(np.diff(peak_times))))
Q_factor = omega_0 / (-2 * decay_fit[0])
Q_fac_error = abs(Q_factor - (omega_0 / (-2 * (decay_fit[0] - fit_err[0]))))
print("Q-fac:",Q_factor)

print("Q-fac Error:",Q_fac_error)
t_fit = np.linspace(-10, np.max(peak_times) + 10, 20)
amplitude_fit = np.exp(np.polyval(decay_fit, t_fit))

plt.figure()
plt.errorbar(
    peak_times,
    peak_amplitudes,
    fmt="o",
    xerr=0,
    yerr=b_error,
    markersize=4,
    alpha=0.5,
    label="Data",
)
plt.plot(t_fit, amplitude_fit, "--", label="Exponential Fit ($Q = "+str(int(Q_factor)) +"\\pm"+str(int(Q_fac_error*10)/10) +"$)")

plt.fill_between(
    t_fit,
    np.exp(np.polyval(decay_fit + fit_err, t_fit)),
    y2=np.exp(np.polyval(decay_fit - fit_err, t_fit)),
    alpha=0.2,
)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude ($rad$)")
plt.ylim(np.min(peak_amplitudes) * 0.95, np.max(theta) * 1.05)
plt.xlim(-10, np.max(peak_times) + 10)
plt.grid(alpha=0.5)
plt.legend()
plt.show()



## PERIOD AMPLITUDE ##

periods = np.concatenate((np.diff(time[peaks]), np.diff(time[peaks_b])))
plt.plot(periods)
plt.show()
amplitudes = np.concatenate((theta[peaks[:-1]], theta[peaks_b[:-1]]))


print("Period:", np.average(periods))
print("Error:", np.std(periods)/np.sqrt(len(periods)))
