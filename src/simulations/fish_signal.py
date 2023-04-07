import sys
from IPython import embed
import thunderfish.powerspectrum as ps
import numpy as np

species_name = dict(
    Sine="Sinewave",
    Alepto="Apteronotus leptorhynchus",
    Arostratus="Apteronotus rostratus",
    Eigenmannia="Eigenmannia spec.",
    Sternarchella="Sternarchella terminalis",
    Sternopygus="Sternopygus dariensis",
)
"""Translate species ids used by wavefish_harmonics and pulsefish_eodpeaks to full species names.
"""


def abbrv_genus(name):
    """Abbreviate genus in a species name.

    Parameters
    ----------
    name: string
        Full species name of the form 'Genus species subspecies'

    Returns
    -------
    name: string
        The species name with abbreviated genus, i.e. 'G. species subspecies'
    """
    ns = name.split()
    return ns[0][0] + ". " + " ".join(ns[1:])


# Amplitudes and phases of various wavefish species:

Sine_harmonics = dict(amplitudes=(1.0,), phases=(0.5 * np.pi,))

Apteronotus_leptorhynchus_harmonics = dict(
    amplitudes=(0.90062, 0.15311, 0.072049, 0.012609, 0.011708),
    phases=(1.3623, 2.3246, 0.9869, 2.6492, -2.6885),
)

Apteronotus_rostratus_harmonics = dict(
    amplitudes=(
        0.64707,
        0.43874,
        0.063592,
        0.07379,
        0.040199,
        0.023073,
        0.0097678,
    ),
    phases=(2.2988, 0.78876, -1.316, 2.2416, 2.0413, 1.1022, -2.0513),
)

Eigenmannia_harmonics = dict(
    amplitudes=(1.0087, 0.23201, 0.060524, 0.020175, 0.010087, 0.0080699),
    phases=(1.3414, 1.3228, 2.9242, 2.8157, 2.6871, -2.8415),
)

Sternarchella_terminalis_harmonics = dict(
    amplitudes=(
        0.11457,
        0.4401,
        0.41055,
        0.20132,
        0.061364,
        0.011389,
        0.0057985,
    ),
    phases=(-2.7106, 2.4472, 1.6829, 0.79085, 0.119, -0.82355, -1.9956),
)

Sternopygus_dariensis_harmonics = dict(
    amplitudes=(
        0.98843,
        0.41228,
        0.047848,
        0.11048,
        0.022801,
        0.030706,
        0.019018,
    ),
    phases=(1.4153, 1.3141, 3.1062, -2.3961, -1.9524, 0.54321, 1.6844),
)

wavefish_harmonics = dict(
    Sine=Sine_harmonics,
    Alepto=Apteronotus_leptorhynchus_harmonics,
    Arostratus=Apteronotus_rostratus_harmonics,
    Eigenmannia=Eigenmannia_harmonics,
    Sternarchella=Sternarchella_terminalis_harmonics,
    Sternopygus=Sternopygus_dariensis_harmonics,
)
"""Amplitudes and phases of EOD waveforms of various species of wave-type electric fish."""


def wavefish_spectrum(fish):
    """Amplitudes and phases of a wavefish EOD.

    Parameters
    ----------
    fish: string, dict or tuple of lists/arrays
        Specify relative amplitudes and phases of the fundamental and its harmonics.
        If string then take amplitudes and phases from the `wavefish_harmonics` dictionary.
        If dictionary then take amplitudes and phases from the 'amlitudes' and 'phases' keys.
        If tuple then the first element is the list of amplitudes and
        the second one the list of relative phases in radians.

    Returns
    -------
    amplitudes: array of floats
        Amplitudes of the fundamental and its harmonics.
    phases: array of floats
        Phases in radians of the fundamental and its harmonics.

    Raises
    ------
    KeyError:
        Unknown fish.
    IndexError:
        Amplitudes and phases differ in length.
    """
    if isinstance(fish, (tuple, list)):
        amplitudes = fish[0]
        phases = fish[1]
    elif isinstance(fish, dict):
        amplitudes = fish["amplitudes"]
        phases = fish["phases"]
    else:
        if fish not in wavefish_harmonics:
            raise KeyError(
                "unknown wavefish. Choose one of "
                + ", ".join(wavefish_harmonics.keys())
            )
        amplitudes = wavefish_harmonics[fish]["amplitudes"]
        phases = wavefish_harmonics[fish]["phases"]
    if len(amplitudes) != len(phases):
        raise IndexError("need exactly as many phases as amplitudes")
    # remove NaNs:
    for k in reversed(range(len(amplitudes))):
        if np.isfinite(amplitudes[k]) or np.isfinite(phases[k]):
            amplitudes = amplitudes[: k + 1]
            phases = phases[: k + 1]
            break
    return amplitudes, phases


def wavefish_eods(
    fish="Eigenmannia",
    frequency=100.0,
    samplerate=44100.0,
    duration=1.0,
    phase0=0.0,
    noise_std=0.05,
):
    """Simulate EOD waveform of a wave-type fish.

    The waveform is constructed by superimposing sinewaves of integral
    multiples of the fundamental frequency - the fundamental and its
    harmonics.  The fundamental frequency of the EOD (EODf) is given by
    `frequency`. With `fish` relative amplitudes and phases of the
    fundamental and its harmonics are specified.

    The generated waveform is `duration` seconds long and is sampled with
    `samplerate` Hertz.  Gaussian white noise with a standard deviation of
    `noise_std` is added to the generated waveform.

    Parameters
    ----------
    fish: string, dict or tuple of lists/arrays
        Specify relative amplitudes and phases of the fundamental and its harmonics.
        If string then take amplitudes and phases from the `wavefish_harmonics` dictionary.
        If dictionary then take amplitudes and phases from the 'amlitudes' and 'phases' keys.
        If tuple then the first element is the list of amplitudes and
        the second one the list of relative phases in radians.
    frequency: float or array of floats
        EOD frequency of the fish in Hertz. Either fixed number or array for
        time-dependent frequencies.
    samplerate: float
        Sampling rate in Hertz.
    duration: float
        Duration of the generated data in seconds. Only used if frequency is scalar.
    phase0: float
        Phase offset of the EOD waveform in radians.
    noise_std: float
        Standard deviation of additive Gaussian white noise.

    Returns
    -------
    data: array of floats
        Generated data of a wave-type fish.

    Raises
    ------
    KeyError:
        Unknown fish.
    IndexError:
        Amplitudes and phases differ in length.
    """
    # get relative amplitude and phases:
    amplitudes, phases = wavefish_spectrum(fish)
    # compute phase:
    if np.isscalar(frequency):
        phase = np.arange(0, duration, 1.0 / samplerate)
        phase *= frequency
    else:
        phase = np.cumsum(frequency) / samplerate
    # generate EOD:
    data = np.zeros(len(phase))
    for har, (ampl, phi) in enumerate(zip(amplitudes, phases)):
        if np.isfinite(ampl) and np.isfinite(phi):
            data += ampl * np.sin(
                2 * np.pi * (har + 1) * phase + phi - (har + 1) * phase0
            )
    # add noise:
    data += noise_std * np.random.randn(len(data))
    return data


def normalize_wavefish(fish, mode="peak"):
    """Normalize amplitudes and phases of wave-type EOD waveform.

    The amplitudes and phases of the Fourier components are adjusted
    such that the resulting EOD waveform has a peak-to-peak amplitude
    of two and the peak of the waveform is at time zero (mode is set
    to 'peak') or that the fundamental has an amplitude of one and a
    phase of 0 (mode is set to 'zero').

    Parameters
    ----------
    fish: string, dict or tuple of lists/arrays
        Specify relative amplitudes and phases of the fundamental and its harmonics.
        If string then take amplitudes and phases from the `wavefish_harmonics` dictionary.
        If dictionary then take amplitudes and phases from the 'amlitudes' and 'phases' keys.
        If tuple then the first element is the list of amplitudes and
        the second one the list of relative phases in radians.
    mode: 'peak' or 'zero'
        How to normalize amplitude and phases:
        - 'peak': normalize waveform to a peak-to-peak amplitude of two
          and shift it such that its peak is at time zero.
        - 'zero': scale amplitude of fundamental to one and its phase to zero.

    Returns
    -------
    amplitudes: array of floats
        Adjusted amplitudes of the fundamental and its harmonics.
    phases: array of floats
        Adjusted phases in radians of the fundamental and its harmonics.

    """
    # get relative amplitude and phases:
    amplitudes, phases = wavefish_spectrum(fish)
    if mode == "zero":
        newamplitudes = np.array(amplitudes) / amplitudes[0]
        newphases = np.array(
            [p + (k + 1) * (-phases[0]) for k, p in enumerate(phases)]
        )
        newphases %= 2.0 * np.pi
        newphases[newphases > np.pi] -= 2.0 * np.pi
        return newamplitudes, newphases
    else:
        # generate waveform:
        eodf = 100.0
        rate = 100000.0
        data = wavefish_eods(fish, eodf, rate, 2.0 / eodf, noise_std=0.0)
        # normalize amplitudes:
        ampl = 0.5 * (np.max(data) - np.min(data))
        newamplitudes = np.array(amplitudes) / ampl
        # shift phases:
        deltat = np.argmax(data[: int(rate / eodf)]) / rate
        deltap = 2.0 * np.pi * deltat * eodf
        newphases = np.array(
            [p + (k + 1) * deltap for k, p in enumerate(phases)]
        )
        newphases %= 2.0 * np.pi
        newphases[newphases > np.pi] -= 2.0 * np.pi
        # return:
        return newamplitudes, newphases


def export_wavefish(fish, name="Unknown_harmonics", file=None):
    """Serialize wavefish parameter to python code.

    Add output to the wavefish_harmonics dictionary!

    Parameters
    ----------
    fish: string, dict or tuple of lists/arrays
        Specify relative amplitudes and phases of the fundamental and its harmonics.
        If string then take amplitudes and phases from the `wavefish_harmonics` dictionary.
        If dictionary then take amplitudes and phases from the 'amlitudes' and 'phases' keys.
        If tuple then the first element is the list of amplitudes and
        the second one the list of relative phases in radians.
    name: string
        Name of the dictionary to be written.
    file: string or file or None
        File name or open file object where to write wavefish dictionary.

    Returns
    -------
    fish: dict
        Dictionary with amplitudes and phases.
    """
    # get relative amplitude and phases:
    amplitudes, phases = wavefish_spectrum(fish)
    # write out dictionary:
    if file is None:
        file = sys.stdout
    try:
        file.write("")
        closeit = False
    except AttributeError:
        file = open(file, "w")
        closeit = True
    n = 6
    file.write(name + " = \\\n")
    file.write("    dict(amplitudes=(")
    file.write(", ".join(["%.5g" % a for a in amplitudes[:n]]))
    for k in range(n, len(amplitudes), n):
        file.write(",\n")
        file.write(" " * (9 + 12))
        file.write(", ".join(["%.5g" % a for a in amplitudes[k : k + n]]))
    file.write("),\n")
    file.write(" " * 9 + "phases=(")
    file.write(", ".join(["%.5g" % p for p in phases[:n]]))
    for k in range(n, len(phases), n):
        file.write(",\n")
        file.write(" " * (9 + 8))
        file.write(", ".join(["%.5g" % p for p in phases[k : k + n]]))
    file.write("))\n")
    if closeit:
        file.close()
    # return dictionary:
    harmonics = dict(amplitudes=amplitudes, phases=phases)
    return harmonics


def chirps(
    eodf=100.0,
    samplerate=44100.0,
    duration=1.0,
    chirp_times=[0.5],
    chirp_size=[100.0],
    chirp_width=[0.01],
    chirp_kurtosis=[1.0],
    chirp_contrast=[0.05],
):
    """Simulate frequency trace with chirps.

    A chirp is modeled as a Gaussian frequency modulation.
    The first chirp is placed at 0.5/chirp_freq.

    Parameters
    ----------
    eodf: float
        EOD frequency of the fish in Hertz.
    samplerate: float
        Sampling rate in Hertz.
    duration: float
        Duration of the generated data in seconds.
    chirp_times: float
        Timestamps of every single chirp in seconds.
    chirp_size: list
        Size of each chirp (maximum frequency increase above eodf) in Hertz.
    chirp_width: list
        Width of every single chirp at 10% height in seconds.
    chirp_kurtosis: list:
        Shape of every single chirp. =1: Gaussian, >1: more rectangular, <1: more peaked.
    chirp_contrast: float
        Maximum amplitude reduction of EOD during every respective chirp.

    Returns
    -------
    frequency: array of floats
        Generated frequency trace that can be passed on to wavefish_eods().
    amplitude: array of floats
        Generated amplitude modulation that can be used to multiply the trace generated by
        wavefish_eods().
    """
    # baseline eod frequency and amplitude modulation:
    n = len(np.arange(0, duration, 1.0 / samplerate))
    frequency = eodf * np.ones(n)
    am = np.ones(n)

    for time, width, size, kurtosis, contrast in zip(chirp_times, chirp_width, chirp_size, chirp_kurtosis, chirp_contrast):

        # chirp frequency waveform:
        chirp_t = np.arange(-2.0 * width, 2.0 * width, 1.0 / samplerate)
        chirp_sig = (
            0.5 * width / (2.0 * np.log(10.0)) ** (0.5 / kurtosis)
        )
        gauss = np.exp(-0.5 * ((chirp_t / chirp_sig) ** 2.0) ** kurtosis)


        # add chirps on baseline eodf:
        index = int(time * samplerate)
        i0 = index - len(gauss) // 2
        i1 = i0 + len(gauss)
        gi0 = 0
        gi1 = len(gauss)
        if i0 < 0:
            gi0 -= i0
            i0 = 0
        if i1 >= len(frequency):
            gi1 -= i1 - len(frequency)
            i1 = len(frequency)
        frequency[i0:i1] += size * gauss[gi0:gi1]
        am[i0:i1] -= contrast * gauss[gi0:gi1]

    return frequency, am


def rises(
    eodf,
    samplerate,
    duration,
    rise_times,
    rise_size,
    rise_tau,
    decay_tau,
):
    """Simulate frequency trace with rises.

    A rise is modeled as a double exponential frequency modulation.

    Parameters
    ----------
    eodf: float
        EOD frequency of the fish in Hertz.
    samplerate: float
        Sampling rate in Hertz.
    duration: float
        Duration of the generated data in seconds.
    rise_times: list 
        Timestamp of each of the rises in seconds.
    rise_size: list
        Size of the respective rise (frequency increase above eodf) in Hertz.
    rise_tau: list
        Time constant of the frequency increase of the respective rise in seconds.
    decay_tau: list
        Time constant of the frequency decay of the respective rise in seconds.

    Returns
    -------
    data: array of floats
        Generate frequency trace that can be passed on to wavefish_eods().
    """
    n = len(np.arange(0, duration, 1.0 / samplerate))

    # baseline eod frequency:
    frequency = eodf * np.ones(n)

    for time, size, riset, decayt in zip(rise_times, rise_size, rise_tau, decay_tau):  

        # rise frequency waveform:
        rise_t = np.arange(0.0, 5.0 * decayt, 1.0 / samplerate)
        rise = (
            size
            * (1.0 - np.exp(-rise_t / riset))
            * np.exp(-rise_t / decayt)
        )

        # add rises on baseline eodf:
        index = int(time * samplerate)
        if index + len(rise) > len(frequency):
            rise_index = len(frequency) - index
            frequency[index : index + rise_index] += rise[:rise_index]
            break
        else:
            frequency[index : index + len(rise)] += rise
    return frequency

class FishSignal:
    def __init__(self, samplerate, duration, eodf, nchirps, nrises):
        time = np.arange(0, duration, 1 / samplerate)
        chirp_times = np.random.uniform(0, duration, nchirps)
        rise_times = np.random.uniform(0, duration, nrises)

        # pick random parameters for chirps 
        chirp_size = np.random.uniform(60, 200, nchirps)
        chirp_width = np.random.uniform(0.01, 0.1, nchirps)
        chirp_kurtosis = np.random.uniform(1, 1, nchirps)
        chirp_contrast = np.random.uniform(0.1, 0.5, nchirps)

        # pick random parameters for rises
        rise_size = np.random.uniform(10, 100, nrises)
        rise_tau = np.random.uniform(0.5, 1.5, nrises)
        decay_tau = np.random.uniform(5, 15, nrises)

        # generate frequency trace with chirps
        chirp_trace, chirp_amp = chirps(
            eodf=0.0,
            samplerate=samplerate,
            duration=duration,
            chirp_times=chirp_times,
            chirp_size=chirp_size,
            chirp_width=chirp_width,
            chirp_kurtosis=chirp_kurtosis,
            chirp_contrast=chirp_contrast,
        )

        # generate frequency trace with rises
        rise_trace = rises(
            eodf=0.0,
            samplerate=samplerate,
            duration=duration,
            rise_times=rise_times,
            rise_size=rise_size,
            rise_tau=rise_tau,
            decay_tau=decay_tau,
        )

        # combine traces to one
        full_trace = chirp_trace + rise_trace + eodf

        # make the EOD from the frequency trace
        fish = wavefish_eods(
            fish="Alepto",
            frequency=full_trace,
            samplerate=samplerate,
            duration=duration,
            phase0=0.0,
            noise_std=0.05,
        )

        signal = fish * chirp_amp

        self.signal = signal
        self.trace = full_trace
        self.time = time
        self.samplerate = samplerate
        self.eodf = eodf

    def visualize(self):

        spec, freqs, spectime = ps.spectrogram(
            data=self.signal,
            ratetime=self.samplerate,
            freq_resolution=0.5,
            overlap_frac=0.5,
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[1, 4], sharex=True)

        ax1.plot(self.time, self.signal)
        ax1.set_ylabel("Amplitude")
        ax1.set_xlabel("Time (s)")
        ax1.set_title("EOD signal")

        ax2.imshow(ps.decibel(spec), origin='lower', aspect="auto", extent=[spectime[0], spectime[-1], freqs[0], freqs[-1]])
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_xlabel("Time (s)")
        ax2.set_title("Spectrogram")
        ax2.set_ylim(0, 2000)
        plt.show()
