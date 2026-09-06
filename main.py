from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import numpy as np
import scipy


def listen(freq, BW, samps, num):
    sdr = RtlSdr()
    sdr.sample_rate = BW
    sdr.center_freq = freq
    sdr.freq_correction = 60  # PPM
    print(sdr.valid_gains_db)
    sdr.gain = 48.0
    print(sdr.gain)

    x = sdr.read_samples(2048) # get rid of initial empty samples
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




def main():
    num = 500
    freq = np.int64(100_700_000)
    BW = 2.4e6
    samps = np.int64(2048)
    data = listen(freq, BW, samps, num)
    create_plot(data, freq, BW, num, samps)


main()