import numpy as np
import matplotlib.pyplot as plt
import smplotlib  # noqa


lengths = np.array(
    [
        0.2674,
        0.3274,
        0.3624,
        #    0.4137,
        0.4650,
        0.584,
        0.745,
    ]
)
lengths_err = np.array(
    [
        0.002,
        0.001,
        0.0013,
        #    0.0007,
        0.0011,
        0.0012,
        0,
    ]
)
periods = np.array(
    [
        1.02,
        1.12,
        1.1946,
        #    1.294,
        1.372,
        1.525,
        1.707,
    ]
)
period_err = np.array(
    [
        0.0011,
        0.0004,
        0.0002,
        0.0002,
        0.0002,
        0.0005,
    ]
)
qfact = np.array(
    [
        160,
        183,
        189,
        203,
        207,
        197,
    ]
)
qfact_counting = np.array(
    [
        158,
        185,
        196,
        208,
        212,
        198,
    ]
)
qfact_err = np.array(
    [
        2.42,
        0.95,
        1.178,
        0.31,
        0.31,
        0.87,
    ]
)

plt.errorbar(
    lengths,
    periods,
    fmt="o",
    yerr=period_err,
    xerr=0.01,
    markersize=4,
    alpha=0.5,
    label="Data",
)

coeffs, cov = np.polyfit(np.log(lengths), np.log(periods), 1, cov=True)
fit_err = np.sqrt(np.diag(cov))

a = np.exp(coeffs[1])
b = coeffs[0]
x_fit = np.linspace(lengths.min(), lengths.max(), 100)
y_fit = a * x_fit**b

plt.plot(
    x_fit,
    y_fit,
    "--",
    label="Power Fit\nT = $("
    + str(np.round(a, decimals=2))
    + "0\\pm"
    + str(np.round(a - np.exp(coeffs[1] - fit_err[1]), decimals=2))
    + ")L^{"
    + str(np.round(b, decimals=2))
    + "\\pm"
    + str(np.round(fit_err[0], decimals=2))
    + "}$",
)

plt.xlabel("Length ($m$)")
plt.ylabel("Period ($s$)")
#plt.xscale("log")
#plt.yscale("log")
plt.grid(which="both", alpha=0.1)

plt.legend(loc=2, frameon=True)
plt.show()

plt.errorbar(
    lengths,
    qfact,
    fmt="o",
    xerr=0.01,
    yerr=qfact_err,
    markersize=4,
    alpha=0.5,
    label="$\\tau$ Fitting Method",
)
#plt.scatter(lengths, qfact_counting, marker="x", label="Q-Counting Method")

fit, cov = np.polyfit(lengths, qfact, 2, cov=True)
err = np.sqrt(np.diag(cov))

x_fit = np.linspace(0, 1.2, 100)
y_fit = np.polyval(fit, x_fit)
plt.plot(
    x_fit,
    y_fit,
    "--",
    label="Fit: $Q = aL^2+bL+c$\n$a:"
    + str(int(fit[0]))
    + "\\pm"
    + str(int(err[0] / 10) * 10)
    + "$\n$b:"
    + str(int(fit[1]/10)*10)
    + "\\pm"
    + str(int(err[1] / 10) * 10)
    + "$\n$c:"
    + str(int(fit[2]))
    + "\\pm"
    + str(int(err[2] / 10) * 10)
    + "$",
)
plt.xlim(0.2, 0.8)
plt.ylim(100, 250)
plt.xlabel("Length (m)")
plt.ylabel("Q-Factor")
plt.grid(alpha=0.5)
plt.legend(frameon=True)
plt.show()
