import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from scipy.stats import norm


def make_positions(origin, boundaries, trajectories, steps):
    x = np.full(len(trajectories) + 1, np.nan)
    y = np.full(len(trajectories) + 1, np.nan)
    x[0] = origin[0]
    y[0] = origin[1]

    for i in range(len(trajectories)):
        # use the fist trajectory as is
        if i == 0:
            converted_trajectory = trajectories[i]

        # make all othes trajectories relative to the previous one
        else:
            converted_trajectory = trajectories[i - 1] - trajectories[i]

            # make sure the trajectory is between 0 and 2pi
            if converted_trajectory > 2 * np.pi:
                converted_trajectory = converted_trajectory - 2 * np.pi
            if converted_trajectory < 0:
                converted_trajectory = converted_trajectory + 2 * np.pi

            # write current trajectory to trajectories to correct
            # future trajectories relative to the current one
            trajectories[i] = converted_trajectory

        # use trigonometric identities to calculate the x and y positions
        y[i + 1] = np.sin(converted_trajectory) * steps[i]
        x[i + 1] = np.cos(converted_trajectory) * steps[i]

    # cumulatively add the steps to the positions
    x = np.cumsum(x)
    y = np.cumsum(y)

    # fold back the positions if they are outside the boundaries
    boundaries = np.ravel(boundaries)
    while (
        np.any(x < boundaries[0])
        or np.any(x > boundaries[1])
        or np.any(y < boundaries[2])
        or np.any(y > boundaries[3])
    ):
        x[x < boundaries[0]] = boundaries[0] + (
            boundaries[0] - x[x < boundaries[0]]
        )
        x[x > boundaries[1]] = boundaries[1] - (
            x[x > boundaries[1]] - boundaries[1]
        )
        y[y < boundaries[2]] = boundaries[2] + (
            boundaries[2] - y[y < boundaries[2]]
        )
        y[y > boundaries[3]] = boundaries[3] - (
            y[y > boundaries[3]] - boundaries[3]
        )

    return x, y


class FishMovement:
    def __init__(self, duration, target_fs, origin, boundaries):
        self.duration = duration
        self.fs = target_fs
        self.origin = origin
        self.boundaries = boundaries

        fs = 30  # sampling frequency in Hz
        peak_veloc = 0.2  # most common velocity in m/s
        self.directions = np.arange(
            0, 2 * np.pi, 0.0001
        )  # directions vector in radians

        # make probablitiy distribution of directions
        sigma = 7 / fs
        p1 = norm.pdf(self.directions, 0, sigma)
        p2 = norm.pdf(self.directions, np.max(self.directions), sigma)
        probabilities = p1 + p2
        self.probabilities = probabilities / np.sum(probabilities)

        # make random step lengths according to a gamma distribution
        step_lengths = (
            np.random.default_rng().gamma(
                peak_veloc * 100, 1, (self.duration * fs) - 1
            )
            / 100
        )

        # remove outliers
        step_lengths[step_lengths > 1] = 1

        # normalize to sampling rate
        self.step_lengths = step_lengths / fs

        # draw random directions according to the probability distribution
        self.trajectories = np.random.choice(
            self.directions, size=(self.duration * fs) - 1, p=self.probabilities
        )

        # make positions
        self._x, self._y = make_positions(
            self.origin, self.boundaries, self.trajectories, self.step_lengths
        )

        # resample to target sampling rate
        self.x = np.interp(
            np.arange(0, duration, 1 / target_fs),
            np.arange(0, duration, 1 / fs),
            self._x,
        )
        self.y = np.interp(
            np.arange(0, duration, 1 / target_fs),
            np.arange(0, duration, 1 / fs),
            self._y,
        )
        self.original_fs = fs

    def vizualize(self):
        fig = plt.figure(layout="constrained")
        gs0 = gridspec.GridSpec(1, 2, figure=fig)
        gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[0])
        ax1 = fig.add_subplot(gs00[0])
        ax2 = fig.add_subplot(gs00[1], projection="polar")
        ax3 = fig.add_subplot(gs0[1])
        ax3.set_aspect("equal")

        ax1.hist(self.step_lengths * self.original_fs, bins=100)
        ax1.set_xlabel("Velocities (m/s)")
        ax1.set_ylabel("Count")
        ax1.set_title("Velocity distribution")

        ax2.set_theta_zero_location("N")
        ax2.plot(self.directions, self.probabilities)
        ax2.set_rlabel_position(
            -22.5
        )  # Move radial labels away from plotted line
        ax2.set_title(
            f"Probability distribution of possible heading directions for an average {np.mean(self.step_lengths*100)} cm step"
        )

        ax3.plot(self._x, self._y)
        ax3.set_xlabel("x (m)")
        ax3.set_ylabel("y (m)")
        ax3.set_title(
            f"Simulated positions over {self.duration} seconds at {self.original_fs} Hz converted to {self.fs} Hz"
        )
        plt.show()


if __name__ == "__main__":
    tmax = 6000  # duration in seconds
    origin = [0, 0]  # starting point
    boundaries = [(0, 10), (0, 10)]  # boundaries of the arena
    m = FishMovement(tmax, 100, origin, boundaries)
    m.vizualize()
