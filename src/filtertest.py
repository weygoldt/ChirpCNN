import matplotlib.pyplot as plt
import numpy as np

# Generate example signal: sinusoid with changing frequency and noise
np.random.seed(0)
t = np.linspace(0, 5, num=500)  # Time vector
f0 = 500  # Initial frequency
f1 = 800  # Final frequency
signal = 0.5 * np.sin(
    2 * np.pi * (f0 + (f1 - f0) * t / 5) * t
) + 0.2 * np.random.randn(500)

# Define state space model parameters
dt = 1  # Time step
A = np.array([[1, dt], [0, 1]])  # State transition matrix
B = np.array([[0.5 * dt**2], [dt]])  # Control input matrix
H = np.array([[1, 0]])  # Observation matrix
Q = np.array([[1e-6, 0], [0, 1e-6]])  # State noise covariance matrix
R = np.array([[0.1]])  # Observation noise covariance matrix

# Initialize the Kalman filter
state_estimate = np.array(
    [500, 0]
)  # Initial state estimate (frequency, frequency rate)
state_covariance = np.eye(2)  # Initial state covariance
filtered_frequency = np.zeros_like(signal)  # Filtered frequency estimates

# Loop through each time point and update the filter
for i in range(signal.shape[0]):
    # Extract input signal for current time point
    input_signal = signal[i]

    # Update the state estimate and covariance
    state_estimate = A @ state_estimate + B @ state_estimate[1]
    state_covariance = A @ state_covariance @ A.T + Q

    # Calculate the Kalman gain
    kalman_gain = (
        state_covariance @ H.T @ np.linalg.inv(H @ state_covariance @ H.T + R)
    )

    # Update the state estimate based on the observation
    innovation = input_signal - H @ state_estimate
    state_estimate = state_estimate + kalman_gain @ innovation

    # Update the state covariance
    state_covariance = (np.eye(2) - kalman_gain @ H) @ state_covariance

    # Extract the filtered frequency estimate for current time point
    filtered_frequency[i] = state_estimate[0]

# Plot the original signal, filtered frequency, and true frequency
plt.figure(figsize=(12, 6))
plt.plot(t, signal, label="Original Signal")
plt.plot(t, filtered_frequency, label="Filtered Frequency")
plt.plot(t, f0 + (f1 - f0) * t / 5, label="True Frequency")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Kalman Filter for Time-Varying Frequency Extraction")
plt.show()
