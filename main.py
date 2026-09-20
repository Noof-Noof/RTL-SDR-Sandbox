from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sounddevice as sd


def listen(freq, BW, samps, num):
    sdr = RtlSdr()
    sdr.sample_rate = BW
    sdr.center_freq = freq
    sdr.freq_correction = 60  # PPM
    print(sdr.valid_gains_db)
    sdr.gain = 48.0
    print(sdr.gain)

    x = sdr.read_samples(2048)
    x = sdr.read_samples(samps*num)
    sdr.close()
    return x


def create_plot(data, center_freq, sample_rate, num, samps):
    plt.plot(data.real)
    plt.plot(data.imag)
    plt.show()
    fft_size = samps
    num_rows = num
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(data[i*fft_size:(i+1)*fft_size])))**2)
    extent = [(center_freq + sample_rate/   -2)/1e6,
                (center_freq + sample_rate/2)/1e6,
                len(data)/sample_rate, 0]
    plt.imshow(spectrogram, aspect='auto', extent=extent)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Time [s]")
    plt.show()

def fm_demodulate(iq_samples, sample_rate, audio_rate=48000):
    iq_samples = np.asarray(iq_samples, dtype=np.complex64)
    prod = iq_samples[1:] * np.conj(iq_samples[:-1])
    demod = np.angle(prod)  # radians, range [-pi, pi]
    stage1_factor = int(sample_rate // 240000)
    if stage1_factor > 1:
        demod = scipy.signal.decimate(demod, stage1_factor, ftype='fir', zero_phase=True)
        intermediate_rate = sample_rate / stage1_factor
    else:
        intermediate_rate = sample_rate

    stage2_factor = int(round(intermediate_rate / audio_rate))
    if stage2_factor > 1:
        factor = stage2_factor
        while factor > 1:
            f = min(factor, 10)
            demod = scipy.signal.decimate(demod, f, ftype='fir', zero_phase=True)
            factor //= f
    demod = demod / (np.max(np.abs(demod)) + 1e-9)

    return demod.astype(np.float32), int(audio_rate)

def play_audio(audio_samples, sample_rate=48000):
    audio_samples = np.asarray(audio_samples, dtype=np.float32)
    sd.play(audio_samples, samplerate=sample_rate, blocking=True)


def main():
    num = 500
    freq = np.int64(91_900_000)
    BW = 2.4e6
    samps = np.int64(2**14)
    data = listen(freq, BW, samps, num)
    create_plot(data, freq, BW, num, samps)
    demod, rate = fm_demodulate(data, BW)
    play_audio(demod, rate)


main()