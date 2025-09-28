from typing import List
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sgn
import smplotlib
from numpy import genfromtxt
from scipy.interpolate import interp1d
import numpy.linalg as la

my_data = genfromtxt('data2.csv', delimiter=',')
#my_data2 = genfromtxt('data.csv', delimiter=',')

my_data = my_data[140:]
my_data[:,0] -= my_data[0,0] # Set time baseline

length = 0.745 # length in m for trig stuff lol
# angle and energy
fig, (ax1,ax2) = plt.subplots(2)
time = my_data[:,0]
x_displacement = my_data[:,1]
x_displacement -= np.average(x_displacement) # Remove Assymetry
x_displacement_filtered = sgn.savgol_filter(x_displacement, 21, 3)
rate = 1000  # points per second
# Interpolate data
f = interp1d(time, x_displacement_filtered, kind='cubic', bounds_error=False, fill_value='extrapolate')
time = np.linspace(0, time.max(), int(time.max() * rate))
dt = 1/rate  # time step
x_displacement = f(time)
peaks, _ = sgn.find_peaks(x_displacement)
peaks_b, _ = sgn.find_peaks(-x_displacement)

theta = np.arcsin(x_displacement/length)

ax1.plot(time, theta)
ax1.plot(my_data[:,0],np.arcsin(my_data[:,1]/length),'--')
ax1.plot(time[peaks], theta[peaks], 'ro', markersize=6, label='Peaks')
ax1.plot(time[peaks_b], theta[peaks_b], 'ro', markersize=6, label='Peaks')

ax1.set_ylabel("Theta ($\\Theta$)")
ax1.set_xlabel("Time (s)")
ax1.legend()

mg = 9.81*length-np.cos(theta)*9.81*length
ke = 0.5*((np.gradient(theta)/dt)*length)**2
ax2.plot(time,1000*(mg+ke))
ax2.set_ylabel("Energy($mJ/kg$)")
ax2.set_xlabel("Time (s)")

plt.tight_layout()
plt.show()

## FOURIER TRANSFORM ##
freq = np.fft.fftfreq(len(theta), dt)
fft_x = np.fft.fft(x_displacement)

plt.figure()
freq = freq[:len(freq)//2]
fft_x = np.abs(fft_x[:len(fft_x)//2])
plt.fill_between(freq,fft_x,fc='#FF000040',ec='black')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (arb. units)")
plt.ylim(0,np.max(fft_x)*1.1)
plt.xlim(0,6)
plt.title("FFT")
plt.grid(alpha=0.5)
plt.show()

## Amplitude decay
# Exponential decay fit for Q-factor calculation
peak_times = time[peaks]
peak_amplitudes = np.abs(theta[peaks])

# Fit exponential decay: A(t) = A0 * exp(-t/tau)
# Taking log: ln(A) = ln(A0) - t/tau
# Linear fit to get decay constant
decay_fit = np.polyfit(peak_times, np.log(peak_amplitudes), 1)
decay_rate = -decay_fit[0]  # 1/tau

# Calculate Q-factor
# For underdamped oscillator: Q = ω₀/(2γ) where γ = 1/tau
# Approximate natural frequency from FFT peak
dominant_freq_idx = np.argmax(fft_x)
omega_0 = 2 * np.pi * freq[dominant_freq_idx]
Q_factor = omega_0 / (2 * decay_rate)

# Plot the exponential decay
plt.figure()
t_fit = np.linspace(peak_times.min(), peak_times.max(), 100)
amplitude_fit = np.exp(np.polyval(decay_fit, t_fit))

plt.plot(peak_times, peak_amplitudes, 'o', markersize=6, label='Amplitudes')
plt.plot(t_fit, amplitude_fit, '--', label=f'Exponential Fit (Q = {Q_factor:.2f})')
plt.xlabel('Time (s)')
plt.yscale('log')
plt.ylabel('Amplitude ($rad$)')
plt.title('Decay')
plt.grid(alpha=0.5)
plt.legend()
plt.show()

print(f"Q-factor: {Q_factor:.2f}")
print(f"Decay rate: {decay_rate:.4f} s⁻¹")


## PERIOD AMPLITUDE ##

periods = np.concatenate((np.diff(time[peaks]),np.diff(time[peaks_b])))
amplitudes = np.concatenate((theta[peaks[:-1]],theta[peaks_b[:-1]]))

y_error = []
x_error = []
temp_store_periods = []
temp_store_amplitudes = []
i = 0
while i < len(periods):
    done = False
    if i+1 != len(periods):
        if abs(amplitudes[i]-amplitudes[i+1]) < 0.05:
            temp_store_periods.append(periods[i+1])
            temp_store_amplitudes.append(amplitudes[i+1])
            periods = np.delete(periods,i+1)
            amplitudes = np.delete(amplitudes,i+1)
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
        temp_store_amplitudes, temp_store_periods = [],[]
        i += 1
y_error = np.array(y_error)
y_error[y_error == 0] = y_error.max()
fit,cov = np.polyfit(amplitudes,periods,2,cov=True)
print("Fit:",fit)
fit_err = np.sqrt(np.diag(cov))
print("Error:",fit_err)

x_vals = np.linspace(amplitudes.min()*1.2,amplitudes.max()*1.2,300)
plt.figure()
plt.plot(x_vals,np.polyval(fit,x_vals),'--',alpha=0.3,label="Fit Line (Quad.)")
plt.fill_between(x_vals,np.polyval(fit-fit_err,x_vals),y2=np.polyval(fit+fit_err,x_vals),alpha=0.2)

plt.axhline(y=2*np.pi*np.sqrt(length/9.81), color='r', linestyle='--', label='Small-Angle Approximation')
plt.errorbar(amplitudes, periods,yerr=y_error,xerr=x_error,fmt='.',capsize=2,ms=2,c='black')
plt.xlabel("Amplitude ($rad.$)")
plt.ylabel("Period (s)")
plt.xlim(x_vals.min(),x_vals.max())
plt.title("Period vs Amplitude")
plt.grid(alpha=0.5)
plt.legend()
plt.show()

plt.errorbar(amplitudes,periods-np.polyval(fit,amplitudes),yerr=y_error,fmt='.',capsize=2,ms=2,c='black')
plt.axhline(y=0, color='r', linestyle='--', label='')
plt.title("Residuals?? hehe")
plt.xlabel("Amplitude ($rad.$)")
plt.ylabel("Delta Period (s-$s_0$)")
plt.ylim(-0.1,0.1)
plt.show()


