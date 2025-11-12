import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sgn
import smplotlib  # noqa
from numpy import genfromtxt
from scipy.interpolate import interp1d

my_data = genfromtxt("data2.csv", delimiter=",")
# my_data2 = genfromtxt('data.csv', delimiter=',')

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
ax2.set_ylim(0, 6)
ax2.set_xlabel("Time (s)")

plt.tight_layout()
plt.show()
"""
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
"""
## Amplitude decay
# Exponential decay fit for Q-factor calculation
peak_times = time[peaks]
peak_amplitudes = np.abs(theta[peaks])

decay_fit, cov = np.polyfit(peak_times[:10], np.log(peak_amplitudes)[:10], 1, cov=True)
fit_err = np.sqrt(np.diag(cov))
omega_0 = 2 * np.pi * (1 / (np.average(np.diff(peak_times[:10]))))
Q_factor = omega_0 / (-2 * decay_fit[0])
Q_fac_error = abs(Q_factor - (omega_0 / (-2 * (decay_fit[0] - fit_err[0]))))

t_fit = np.linspace(-10, np.max(peak_times) + 10, 20)
amplitude_fit = np.exp(np.polyval(decay_fit, t_fit))



decay_fit_1, cov = np.polyfit(peak_times[15 : len(peak_times) - 1], np.log(peak_amplitudes)[15 : len(peak_times) - 1], 1, cov=True)
fit_err_1 = np.sqrt(np.diag(cov))
omega_0_1 = 2 * np.pi * (1 / (np.average(np.diff(peak_times[:10]))))
Q_factor_1 = int(omega_0_1 / (-2 * decay_fit_1[0]))
Q_fac_error_1 = abs(Q_factor_1 - (omega_0_1 / (-2 * (decay_fit_1[0] - fit_err_1[0]))))

amplitude_fit_1 = np.exp(np.polyval(decay_fit_1, t_fit))

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
plt.plot(t_fit, amplitude_fit, "--", label=f"Exp Fit in $t\\in[0s,25s]$\n($Q = "+str(int(Q_factor*10)/10) +"\\pm"+str(int(Q_fac_error)) +"$)")
plt.plot(
    t_fit,
    amplitude_fit_1,
    "--",
    c="b",
    label=f"Exp Fit in $t\\in[75s,125s]$\n($Q = "+str(int(Q_factor_1)) +"\\pm"+str(int(Q_fac_error_1*10)/10) +"$)",
)

plt.xlabel("Time (s)")
# plt.yscale("log")
plt.ylabel("Amplitude ($rad$)")
plt.ylim(np.min(peak_amplitudes) * 0.95, np.max(theta) * 1.05)
plt.xlim(-10, np.max(peak_times) + 10)
plt.grid(alpha=0.5)
plt.legend()
plt.show()

## Q-FACTOR VS AMPLITUDE ##
'''
# Calculate Q-factor for each amplitude range
q_amplitudes = []
q_factors = []

# Use sliding window approach to calculate local Q-factors
window_size = 20  # Number of consecutive peaks to use for each Q calculation

for i in range(len(peak_amplitudes) - window_size - 15):
    window_times = peak_times[i : i + window_size]
    window_amplitudes = peak_amplitudes[i : i + window_size]

    local_decay_fit = np.polyfit(window_times, np.log(window_amplitudes), 1)

    local_period = np.average(np.diff(window_times))
    local_omega_0 = 2 * np.pi / local_period
    local_Q = local_omega_0 / (-2 * local_decay_fit[0])

    q_amplitudes.append(np.mean(window_times))
    q_factors.append(local_Q)

q_amplitudes = np.array(q_amplitudes)
q_factors = np.array(q_factors)

# Plot Q-factor vs amplitude
plt.figure()
plt.plot(q_amplitudes, q_factors, "o-", markersize=4, alpha=0.7)
# plt.xlabel("Amplitude (rad)")
plt.xlabel("Time (s)")

plt.ylabel("Q-factor")
plt.grid(alpha=0.5)
plt.show()
'''
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
        if abs(amplitudes[i] - amplitudes[i + 1]) < 0.05:
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

# Variance Calculation

error_mean = 0
for i_t, i_a in zip(amplitudes, periods):
    error_mean += (np.polyval(fit, i_t) - i_a) ** 2
error_mean /= 3 * (len(periods) - 3)
error_mean = np.sqrt(error_mean)

x_vals = np.linspace(amplitudes.min() * 1.2, amplitudes.max() * 1.2, 300)
plt.figure()
plt.plot(
    x_vals,
    np.polyval(fit, x_vals),
    "--",
    alpha=0.3,
    label="Quadratic Fit ($T=c+bA+aA^2$)\n$c=1.73\\pm0.0005$\n$b=0.000703\\pm0.0005$\n$a=0.109\\pm0.0007$",
)
plt.fill_between(
    x_vals,
    np.polyval(fit - fit_err, x_vals),
    y2=np.polyval(fit + fit_err, x_vals),
    alpha=0.2,
)

plt.axhline(
    y=2 * np.pi * np.sqrt(length / 9.81),
    color="r",
    linestyle="--",
    label="Given Equation",
)
plt.errorbar(
    amplitudes,
    periods,
    yerr=error_mean,
    xerr=x_error,
    fmt=".",
    capsize=2,
    ms=2,
    c="black",
)
plt.xlabel("Amplitude ($rad.$)")
plt.ylabel("Period (s)")
plt.xlim(x_vals.min(), x_vals.max())
plt.grid(alpha=0.5)
plt.legend()
plt.show()

plt.errorbar(
    amplitudes,
    periods - np.polyval(fit, amplitudes),
    yerr=error_mean,
    fmt=".",
    capsize=2,
    ms=2,
    c="black",
)
plt.axhline(y=0, color="r", linestyle="--", label="")
plt.xlabel("Amplitude ($rad.$)")
plt.grid(alpha=0.5)
plt.ylabel("Delta Period (s-$s_0$)")
plt.ylim(-0.015, 0.015)
plt.show()
