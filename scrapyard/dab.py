import matplotlib.pyplot as plt
import numpy as np
import scipy

def ismember(A, B):
    return [np.sum(a == B) for a in A]


def bi2de(bitarray, order='left-lsb'):
    ### Really basic version of bi2de (left-msb version only)
    xar = np.power(2, range(np.size(bitarray)))
    if (order == 'left-msb'):
        xar = np.power(2, list(reversed(range(np.size(bitarray)))))
    return (np.sum(bitarray * xar))


def getInputBitVector(symbols, idx):
    if (symbols.shape[1] > idx):
        bitVector = symbols[:, idx]
    else:
        bitVector = np.zeros(symbols.shape[0], dtype=int)

    return (bitVector)

def poly2trellis(m, g):
    if (type(m) == int):
        m = np.array([m])

    if (type(m) == list):
        k = np.shape(m)
        nr = 1
    elif (type(m) == np.ndarray):
        nr, k = np.shape(np.matrix(m));
    else:
        print("ERROR: nr must be equal to 1.")
        return (-1)

    ## Convert g to decimal, let oct2dec validate the octal values
    g = [int(str(gg), 8) for gg in g]
    # g = oct2dec (g);

    g = np.matrix(g, dtype=int)

    nr, n = np.shape(np.matrix(g));
    if (nr != k):
        print("ERROR: poly2trellis: G must be a k-by-n octal matrix");
        return (-1)

    ##print("Summary N,K,NR %d %d %d" %(n,k,nr))

    ## Check the ranges of the generators
    ## nu = total number of linear shift registers for all k
    nu = sum(m) - k;

    ## Number of states and input symbols needed to capture the state machine
    nstates = 2 ** nu;
    ninputs = 2 ** k;
    noutputs = 2 ** n;

    t = {"numInputSymbols": ninputs,
         "numOutputSymbols": noutputs,
         "numStates": nstates,
         "nextStates": np.zeros((nstates, ninputs), dtype=int),
         "outputs": np.zeros((nstates, ninputs), dtype=int)}

    statebits = range(nstates);
    statebits = [('{0:0%db}' % (nu)).format(x) for x in statebits];

    states = np.zeros((nstates, k), dtype=int);
    newbit = np.zeros((1, k));
    shifts = np.zeros((1, k));
    offset = 0;
    for i in range(k):
        nu_i = m[i] - 1;
        if (nu_i > 0):
            for kk in range(nstates):
                states[kk, i] = int(statebits[kk][offset:offset + nu_i], 2);

        newbit[i] = (1 << nu_i);
        shifts[i] = offset - 1;
        offset += nu_i;

        if (ismember(g[i][:] >= 2 ** m[i], True) != [0]):
            print("poly2trellis: code size is greater than constraint length")
            return (-1)

        if ((g[i][:] < 2 ** nu_i).all() or not ismember(np.mod(g[i][:], 2), 1) != [0]):
            print("poly2trellis: code size is less than constraint length")
            return (-1)

    ## Generate conversion list of all possible output octal values
    outputs = [('{0:o}').format(x) for x in range(noutputs)];
    # str2num (dec2base (0:noutputs - 1, 8))

    ## Walk the trellis, each row index is state, each column index is input
    for s in range(nstates):
        for ii in range(k):
            ## For each next input bit [0,1] calculate inputs to modulo-2 adder
            ## and the next state for the ith linear shift register, left-shifted
            ## to be combined into the total state.
            state = states[s, i] + np.array([[0], [newbit[ii][0]]], dtype=int)
            nextstate = (np.array(state / 2, dtype=int)) * 2 ** (shifts[i] + 1)
            # for z in state:
            #    nextstate = np.append(nextstate, ((z >> 1) << shifts(i)))
            # nextstate =  ((state >> 1) << shifts(i))

            ## Calculate the modulo-2 sum of the state plus input for each n for
            ## this particular shift register.
            ## The MSB of the output represents the n=1 generator(s)
            out = np.array([[0], [0]])
            for adder in range(n):
                bitres = [('{0:0%db}' % (nu + 1)).format(x[0]) for x in (np.bitwise_and(g[i, adder], state))]
                val = np.mod([x.count('1') for x in bitres], 2)
                out += (np.matrix(val, dtype=int) * 2 ** (n - adder - 1)).T

            ## Accumulate contributions to the trellis for this shift register.
            ## The contribution to each input symbol depends on whether the
            ## (k-i)th bit is 0 or 1.
            bitpos = k - i - 1
            for symbol in range(ninputs):
                idx = (symbol & (1 << bitpos))
                t['nextStates'][s, symbol] += nextstate[idx]
                t['outputs'][s, symbol] = np.bitwise_xor(t['outputs'][s, symbol], out[idx]);

        ## Convert output values to octal representation
        for ii in range(np.size(t['outputs'][s, :])):
            t['outputs'][s, ii] = outputs[t['outputs'][s, ii]];
    return (t)


def convenc(msg, t, punct='', s0=0):
    k = int(np.log2(t['numInputSymbols']));
    n = int(np.log2(t['numOutputSymbols']));

    if (type(msg) == int):
        msg = np.array([msg])

    in_symbols = np.size(msg) / k;
    if (in_symbols != int(in_symbols)):
        print("ERROR: convenc: length of MSG must be a multiple of k")
        return (-1)

    # tranpose = columns(msg ) == 1
    # msg = msg.T
    state = s0
    y = []
    for idx in range(0, np.size(msg), k):
        in_sym = bi2de(msg[idx:idx + k], 'left-msb')
        out_sym = int(str(t['outputs'][state, in_sym]), 8)
        state = t['nextStates'][state, in_sym];
        out_bits = ('{0:0%db}' % (n)).format(out_sym)
        for bb in out_bits:
            y.append(bb)
    y = np.array(y, dtype=int)
    return (y)


def vitdec2(trellis, symbols, tblen, mode='hard', N=2):
    t = trellis
    s = symbols

    numInputBits = int(np.log2(t['numOutputSymbols']))
    numOutputSymbols = t['numInputSymbols']

    if s.shape[0] != numInputBits:
        print("ERROR: our trellis description need %d input bits, i got %d!" % (numInputBits, s.shape[0]))
        return (-1)

    # TODO: error checking for soft/hard mode

    expectedBits = np.zeros((t['outputs'].shape[0], numInputBits, t['outputs'].shape[1]), dtype=int)

    for m in range(t['outputs'].shape[0]):
        for n in range(numOutputSymbols):
            symbol = t['outputs'][m, n]
            for o in range(numInputBits):
                expectedBits[m, o, n] = (N - 1) * np.bitwise_and(symbol >> (numInputBits - o - 1), 1)

    lastStates = np.zeros((t['nextStates'].shape[0], 2), dtype=int) - 1;  # start with -1
    for m in range(t['nextStates'].shape[0]):
        if (lastStates[t['nextStates'][m, 0], 0] < 0):
            lastStates[t['nextStates'][m, 0], 0] = m
        else:
            lastStates[t['nextStates'][m, 0], 1] = m
        if (lastStates[t['nextStates'][m, 1], 0] < 0):
            lastStates[t['nextStates'][m, 1], 0] = m
        else:
            lastStates[t['nextStates'][m, 1], 1] = m
    metric = np.zeros((t['numStates'], tblen + 1), dtype=int)

    bits = []
    dataLen = s.shape[1]
    numInputBits = s.shape[0]
    tbWindowStart = 0
    idx = 0

    inx = np.zeros((symbols.shape[0], t['numStates']), dtype=int)

    while tbWindowStart < dataLen:
        metric = np.roll(metric, -1, axis=1)
        metric[:, -1] = 0

        while idx < tblen:
            inBits = getInputBitVector(s, tbWindowStart + idx)
            for x in range(t['numStates']):
                inx[:, x] = inBits
            for outputSymbol in range(numOutputSymbols):
                state = np.arange(t['numStates'])
                ex0 = expectedBits[state, :, outputSymbol].T
                ns0 = t['nextStates'][state, outputSymbol]
                ns0 = np.array(ns0, dtype=int)
                sm0 = metric[state, idx] + np.sum((N - 1) - np.abs(ex0 - inx), axis=0).T
                metric[ns0[::2], idx + 1] = np.fmax(metric[ns0[::2], idx + 1], sm0[::2])
                metric[ns0[1::2], idx + 1] = np.fmax(metric[ns0[1::2], idx + 1], sm0[1::2])
            idx += 1

        thismetric = metric[:, tblen]
        currentState = (thismetric == max(thismetric)).nonzero()[0].item(0)
        states = np.zeros(tblen + 1)
        states[tblen] = 0
        for idx in np.arange(tblen, 0, -1):
            ls0 = lastStates[currentState, 0]
            ls1 = lastStates[currentState, 1]
            ls0v = metric[ls0, idx - 1]
            ls1v = metric[ls1, idx - 1]
            if (ls0v >= ls1v):
                currentState = ls0
            else:
                currentState = ls1
        if (tbWindowStart > 0):
            bits.append(currentState >> int(np.log2(t['numStates'] - 1)))
        tbWindowStart += 1
        idx = tblen - 1

    return (bits)

def read_complex_byte(data):
    normdata = (np.array(data, dtype=float)-127)/128
    normdata.dtype = complex
    return normdata

def calc_freq_inter():
    prev = 0
    new = 0
    k = 0
    l = 0
    table = np.zeros(1536, dtype=int)
    for i in range(2047): # 2047 interations over 2048 as 0 is predefined
        new = (13 * prev + 511) % 2048
        k = new - 1024
        prev = new
        if (k > -768 and k < 768):
            table[l] = k + 768
            l += 1
    return table

def dab_freq(data, freq, fs, samp):
    # normdata.dtype = complex
    normdata = read_complex_byte(data)
    t = np.arange(0, len(normdata)) / fs
    # frequency shift
    start = 0
    bad = 2000

    trellis = poly2trellis(7, [133, 171, 146, 133])

    plt.plot(abs(normdata[0:samp]))
    plt.title("Synchronisation example for Dab radio")
    plt.show()

    for k in range(len(normdata)):
        n = 0
        if (k+bad) < len(normdata):
            if start == 0:
                for l in range(2000):
                    if np.abs(normdata[k + bad - l]) < 0.2:
                        n += 1
                    else:
                        break
                if n == 2000 and np.abs(normdata[k + bad + 1]) > 0.2:
                    start = k + bad + 1
                    break

    print(start)
    #start = 43221
    Tf = 196608
    Tnull = 2656
    Ts = 2552
    Tu = 2048
    Tg = 504
    K = 1536

    x = normdata[start: start + Tf - Tnull]
    """fs = 2.048e6
    t = np.arange(0, len(x)) / fs
    ff = 110
    x = x * np.exp(1j * 2 * np.pi * ff * t)"""
    symbols = np.reshape(x, (-1, Ts))
    print(symbols.shape)
    # symbol1 = symbols[0, 257:]
    no_cp_symbols = symbols[:, int(Tg/2):-int(Tg/2)]
    print(no_cp_symbols.shape)
    #no_cp_symbols = symbols[Tg:]
    padding = int((Tu-K)/2)
    transform = np.fft.fftshift(np.fft.fft(no_cp_symbols, axis=-1), axes=1)
    qpskx = np.zeros([76, 1536], dtype=complex)
    for l in range(len(qpskx)):
        qpskx[l] = transform[l][padding:-padding]

    n = np.arange(len(qpskx[0]))
    freq = n

    k = calc_freq_inter()
    DiflQPSK = qpskx[1:, :] * np.conj(qpskx[:-1, :])
    DifQPSK = DiflQPSK[:, k]

    plt.plot(freq, np.abs(qpskx[0]), 'b')
    plt.plot(freq, np.abs(qpskx[1]), 'g')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.show()
    DifQPSK = DifQPSK / abs(DifQPSK).max()
    plt.plot(DifQPSK[0].real, DifQPSK[0].imag, "x")
    plt.title("DAB constellation diagram")
    plt.grid()
    plt.show()

    threshold = 0.2
    DifBPSK = np.zeros(len(DifQPSK[0]) * 6, dtype=float)
    finalOut = np.zeros(len(DifBPSK), dtype=int)
    for k in range(3):
        for l in range(len(DifQPSK[0])):
            DifBPSK[k * len(DifQPSK[0]) * 2 + l] = DifQPSK[k][l].real
            DifBPSK[k * len(DifQPSK[0]) * 2+ len(DifQPSK[0]) + l] = DifQPSK[k][l].imag

    print(DifBPSK)
    for m, n in enumerate(DifBPSK):
        if n > 0:
            finalOut[m] = 0
        else:
            finalOut[m] = 1

    plt.plot(finalOut.real, finalOut.imag, "x")
    plt.show()

    #data = qpsk_demodulate(finalOut)
    print(finalOut[:10])
    print(finalOut[100:110])
    print(finalOut[1000:1010])
    #print(data)
    # do puncturing here
    reshaped = np.array([finalOut[:2304], finalOut[2304:2304*2], finalOut[2304*2:2304*3], finalOut[2304*3:]])
    print(reshaped[0][:20])
    data = np.zeros([4, 3096], dtype=int)
    P16 = [1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0]
    P15 = [1,1,1,0,1,1,1,0,1,1,1,0,1,1,0,0]
    for n in range(len(data)):
        if P16[n % 16]:
            data[n - 1] = finalOut[int(n - 1 - n//4)]

    eex = np.hstack((np.reshape(data, (-1, 4)).T, np.zeros((4, 1))))
    print(eex)
    #decoded = vitdec2(trellis, eex, 2, 'soft', 2)
    #print(decoded)