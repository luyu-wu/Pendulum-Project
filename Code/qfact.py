import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sgn
import smplotlib  # noqa
from numpy import genfromtxt
from scipy.interpolate import interp1d

my_data = genfromtxt("qfact.csv", delimiter=",")
my_data = my_data[140:]
my_data[:, 0] -= my_data[0, 0]  # Set time baseline

length = 0.745  # length in m for trig stuff lol
b_error = 0.0075 / (length * 2)
# angle and energy
fig, ax1 = plt.subplots()
time = my_data[:, 0]
x_displacement = my_data[:, 1]
y_displacement = my_data[:, 2]
x_displacement_filtered = sgn.savgol_filter(x_displacement, 21, 3)
y_displacement_filtered = sgn.savgol_filter(y_displacement, 21, 3)
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


## ENERGY GRAPH ##
fig, ax2 = plt.subplots()
mg = 9.8 * length * (1 - np.cos(theta))
ke = 0.5 * ((np.gradient(theta) / dt) * length) ** 2
ax2.plot(time, (mg + ke))
ax2.plot(time, mg, c="b", label="Gravitational", alpha=0.2)
ax2.plot(time, ke, c="r", label="Kinetic", alpha=0.2)
ax2.legend()
ax2.grid(alpha=0.5)
ax2.set_ylabel("Energy($J/kg$)")
ax2.set_ylim(0, 0.075)
ax2.set_xlim(0, np.max(time))
ax2.set_xlabel("Time (s)")

plt.tight_layout()
plt.show()

## FOURIER TRANSFORM ##
freq = np.fft.fftfreq(len(theta), dt)
fft_x = np.fft.fft(x_displacement)

plt.figure()
freq = freq[: len(freq) // 2]
fft_x = np.abs(fft_x[: len(fft_x) // 2])
plt.fill_between(freq, fft_x, fc="#FF000040", ec="black")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (arb. units)")
plt.ylim(0, np.max(fft_x) * 1.1)
plt.xlim(0, 6)
plt.title("FFT")
plt.grid(alpha=0.5)
plt.show()
peak_times = time[peaks]
peak_amplitudes = np.abs(theta[peaks])
decay_fit, cov = np.polyfit(peak_times, np.log(peak_amplitudes), 1, cov=True)
fit_err = np.sqrt(np.diag(cov))

omega_0 = 2 * np.pi * (1 / (np.average(np.diff(peak_times))))
Q_factor = omega_0 / (-2 * decay_fit[0])
Q_fac_error = abs(Q_factor - (omega_0 / (-2 * (decay_fit[0] - fit_err[0]))))
print(Q_fac_error)
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
plt.plot(t_fit, amplitude_fit, "--", label="Exponential Fit ($Q = 202 \\pm 0.9$)")

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

## AMPLITUDE CYCLE NUMBER ##
plt.errorbar(
    1 + np.arange(len(peak_amplitudes)),
    peak_amplitudes,
    fmt="o",
    xerr=0,
    yerr=b_error,
    markersize=4,
    alpha=0.5,
    label="Data",
)
plt.xlabel("Number of Oscillations")
plt.ylabel("Amplitude ($rad$)")
plt.axhline(
    y=np.max(peak_amplitudes) * np.exp(-np.pi / 4),
    color="r",
    linestyle="--",
    label="$e^{-\\pi/4}$",
)
plt.axvline(x=45, ls="--", c="b")
plt.axvline(x=55, ls="--", c="b")

plt.fill_between(range(45, 56), np.ones(11), y2=np.zeros(11), fc="b", alpha=0.2)
plt.ylim(0.04, 0.135)
plt.legend()
plt.show()


## PERIOD AMPLITUDE ##

periods = np.concatenate((np.diff(time[peaks]), np.diff(time[peaks_b])))
amplitudes = np.concatenate((theta[peaks[:-1]], theta[peaks_b[:-1]]))

y_error = []
x_error = []
temp_store_periods = []
temp_store_amplitudes = []
i = 0
while i < len(periods):
    done = False
    if i + 1 != len(periods):
        if abs(amplitudes[i] - amplitudes[i + 1]) < 0.02:
            temp_store_periods.append(periods[i + 1])
            temp_store_amplitudes.append(amplitudes[i + 1])
            periods = np.delete(periods, i + 1)
            amplitudes = np.delete(amplitudes, i + 1)
        else:
            done = True
    else:
        done = True
    if done:
        temp_store_periods.append(periods[i])
        temp_store_amplitudes.append(amplitudes[i])
        y_error.append(np.std(temp_store_periods))
        x_error.append(np.std(temp_store_amplitudes))
        periods[i] = np.average(temp_store_periods)
        amplitudes[i] = np.average(temp_store_amplitudes)
        temp_store_amplitudes, temp_store_periods = [], []
        i += 1
y_error = np.array(y_error)
y_error[y_error == 0] = y_error.max()
fit, cov = np.polyfit(amplitudes, periods, 2, cov=True)

print("Fit:", fit)
fit_err = np.sqrt(np.diag(cov))
print("Error:", fit_err)

x_vals = np.linspace(amplitudes.min() * 1.2, amplitudes.max() * 1.2, 300)
plt.figure()
plt.plot(
    x_vals,
    np.polyval(fit, x_vals),
    "--",
    alpha=0.3,
)
plt.fill_between(
    x_vals,
    np.polyval(fit - fit_err, x_vals),
    y2=np.polyval(fit + fit_err, x_vals),
    alpha=0.2,
)
plt.errorbar(
    amplitudes, periods, yerr=y_error, xerr=x_error, fmt=".", capsize=2, ms=2, c="black"
)
plt.xlabel("Amplitude ($rad.$)")
plt.ylabel("Period (s)")
plt.xlim(x_vals.min(), x_vals.max())
plt.grid(alpha=0.5)
plt.legend()
plt.show()
