import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sgn
import smplotlib
from numpy import genfromtxt
from scipy.interpolate import interp1d

savgol = True

my_data = genfromtxt('data.csv', delimiter=',')
length = 0.748 # length in m for trig stuff lol
# x and y displacement plots
fig, (ax1,ax2) = plt.subplots(2)

time = my_data[:,0]
x_displacement = my_data[:,1]

if savgol:
    x_displacement_filtered = sgn.savgol_filter(x_displacement, 41, 3)

    time_max = time.max()
    rate = 1000  # points per second
    time_interp = np.linspace(0, time_max, int(time_max * rate))

    # Interpolate data
    f = interp1d(time, x_displacement_filtered, kind='cubic', bounds_error=False, fill_value='extrapolate')
    x_displacement_interp = f(time_interp)

    peaks, _ = sgn.find_peaks(x_displacement_interp)
    peak_times_for_plot = time_interp[peaks]
    peak_values_for_plot = x_displacement_interp[peaks]
    ax1.plot(time_interp, x_displacement_interp)

else:
    peaks, _ = sgn.find_peaks(x_displacement)
    peak_times_for_plot = time[peaks]
    peak_values_for_plot = x_displacement[peaks]

ax1.plot(time, x_displacement)
ax1.plot(peak_times_for_plot, peak_values_for_plot, 'ro', markersize=6, label='Peaks')
ax1.set_ylabel("X-Displacement (m)")
ax1.set_xlabel("Time (s)")
ax1.legend()

ax2.plot(my_data[:,0],my_data[:,2])
ax2.set_ylabel("Y-Displacement (m)")
ax2.set_xlabel("Time (s)")

plt.tight_layout()
plt.show()

dt = np.average(np.gradient(time))  # time step
freq = np.fft.fftfreq(len(x_displacement), dt)
fft_x = np.fft.fft(x_displacement)

plt.figure()
freq = freq[:len(freq)//2]
fft_x = np.abs(fft_x[:len(fft_x)//2])
plt.fill_between(freq,fft_x,fc='#FF000040',ec='black')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (arb. units)")
plt.ylim(0,np.max(fft_x)*1.1)
plt.xlim(0,6)
plt.title("data fft")
plt.grid(alpha=0.5)
plt.show()

if len(peaks) > 1:
    if savgol:
        peak_times = time_interp[peaks]
        periods = np.diff(peak_times)
        amplitudes = x_displacement_interp[peaks[:-1]]
    else:
        peak_times = my_data[peaks,0]
        periods = np.diff(peak_times)
        amplitudes = my_data[peaks[:-1],1]

    amplitudes = np.degrees(np.arcsin(amplitudes/length))
    fit = np.polyfit(amplitudes,periods,2)
    x_vals = np.linspace(amplitudes.min()*0.8,amplitudes.max()*1.2,300)
    plt.figure()
    plt.plot(x_vals,np.polyval(fit,x_vals),'--',alpha=0.3,label="Fit Line (Quad.)")
    plt.scatter(amplitudes, periods,s=4)
    plt.xlabel("Amplitude ($\\deg.$)")
    plt.ylabel("Period (s)")
    plt.xlim(x_vals.min(),x_vals.max())
    plt.title("Period vs Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()
