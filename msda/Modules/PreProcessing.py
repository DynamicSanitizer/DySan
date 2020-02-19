# Author : Theo Jourdan #
# 2018
import matplotlib
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.fftpack as fftpack
import scipy
import os
from spectrum import *
from pandas import Series
import glob
import os.path
import random

def listdirectory(path):
    fichier = []
    l = glob.glob(path + '\\*')
    for i in l:
        if os.path.isdir(i):
            fichier.extend(listdirectory(i))
        else:
            fichier.append(i)
    return fichier


#Ici toutes les datas en entree doivent etre N:1 dimension

def open_allfile():
    cur = 0
    id = []
    Xacc = []
    Yacc = []
    Zacc = []
    Xgyro = []
    Ygyro = []
    Zgyro = []
    for element in os.listdir('/Users/theoj/Desktop/Stage5BIM/BDDrecognition/newRawData/'):
        cur +=1
        f = open('/Users/theoj/Desktop/Stage5BIM/BDDrecognition/newRawData/' + element)
        featureArray = []
        lines = f.readlines()
        for line in lines:
            feature_length = len(line.split(" "))
            raw_feature = line.split(" ")
            feature = []
            for index in xrange(feature_length):
                try:
                    feature.append(float(raw_feature[index]))
                except:
                    continue
            featureArray.append(feature)
        data = np.asarray(featureArray)
        if cur<= 61:
            for i in data:
                id.append(cur)
                Xacc.append(i[0])
                Yacc.append(i[1])
                Zacc.append(i[2])
        else :
            for i in data:
                Xgyro.append(i[0])
                Ygyro.append(i[1])
                Zgyro.append(i[2])
    return id, Xacc, Yacc, Zacc, Xgyro, Ygyro, Zgyro

############################### MODIFICATION DU SIGNAL ######################################


def SNR(a, axis=0, ddof=0):
    """
    The signal-to-noise ratio of the input data.
    Returns the signal-to-noise ratio of `a`, here defined as the mean
    divided by the standard deviation.
    Parameters
    ----------
    a : array_like
        An array_like object containing the sample data.
    axis : int or None, optional
        If axis is equal to None, the array is first ravel'd. If axis is an
        integer, this is the axis over which to operate. Default is 0.
    ddof : int, optional
        Degrees of freedom correction for standard deviation. Default is 0.
    Returns
    -------
    s2n : ndarray
        The mean to standard deviation ratio(s) along `axis`, or 0 where the
        standard deviation is 0.
    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def contraction(tab):
    i =0
    tableau = tab
    for j in xrange(len(tableau)):
        i+=1
        if i%2 == 0:
            del tableau[j]
    return tableau


def median_filter(data, kernel):
    dataf = signal.medfilt(data, kernel_size=kernel)
    return dataf

def slidingWindow(data,winSize,step):

    window = []

    numOfWind = ((len(data)-winSize)/step)+1
    for i in range(0,int(numOfWind*step),step):
        window.append(data[i:i+winSize])
    return window


def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def getting_body_jerk(datay, datax): #probleme car on se retrouve avec une valeur de moins en sortie, marche pas encore
    djerk = np.diff(datax)/np.diff(datay)
    return djerk
def magnitude (datax, datay, dataz):
    somme = map(sum, zip(np.multiply(datax,datax),np.multiply(datay,datay),np.multiply(dataz,dataz)))
    magn = [i**0.5 for i in somme]
    return magn

def magnitude_slid (datax, datay, dataz):
    totsomme = []
    for i in xrange(len(datax)):
        somme = []
        for j in xrange(len(datax[i])):
            somme.append(sqrt(datax[i][j] * datax[i][j] + datay[i][j] * datay[i][j] + dataz[i][j] * dataz[i][j]))
        totsomme.append(somme)
    return totsomme

def divide_body_gravi_acc (totalAcc, cutoff, fs, order): #apres avoir fait les deux premiers filtres
    graviAcc = butter_lowpass_filter(totalAcc, cutoff, fs, order)
    bodyAcc = totalAcc - graviAcc
    return bodyAcc, graviAcc

def shift_positif (tab, n): #utilisee dans la methode translation
    realTab = np.real(tab)
    imagTab = np.imag(tab)
    Rconstante = realTab[0]
    Iconstante = imagTab[0]

    RlossData = realTab[len(realTab)-n : len(realTab)]
    IlossData = imagTab[len(imagTab)-n : len(imagTab)]

    Rtab = np.roll(realTab,n)
    Itab = np.roll(imagTab,n)
    for i in xrange(n):
        Rtab[i] = RlossData[i]
        Itab[i] = IlossData[i]
    Rtab[n] = RlossData[len(RlossData)-1]
    Itab[n] = IlossData[len(IlossData)-1]
    Rtab[0] = Rconstante
    Itab[0] = Iconstante

    tab2 = Rtab + Itab * 1j
    return tab2

def shift_positif2 (tab, n): #utilisee dans la methode translation
    realTab = np.real(tab)
    imagTab = np.imag(tab)

    Rconstante = realTab[0]
    Iconstante = imagTab[0]

    PositifR = realTab[0:int(len(realTab)/2)]
    NegatifR = realTab[int(len(realTab)/2): len(realTab)]

    PositifI = imagTab[0:int(len(imagTab)/2)]
    NegatifI = imagTab[int(len(imagTab)/2): len(imagTab)]

    PositifRLoss = PositifR[len(PositifR)-n: len(PositifR)]
    NegatifRLoss = NegatifR[0:n]

    PositifILoss = PositifI[len(PositifI)-n: len(PositifI)]
    NegatifILoss = NegatifI[0:n]


    RtabPositif = np.roll(PositifR,n)
    RtabNegatif = np.roll(NegatifR,-n)
    ItabPositif = np.roll(PositifI,n)
    ItabNegatif = np.roll(NegatifI,-n)


    for i in xrange(n):
        RtabPositif[i] = PositifRLoss[i]
        RtabNegatif[len(NegatifR)-i-1] = NegatifRLoss[::-1][i]
        ItabPositif[i] = PositifILoss[i]
        ItabNegatif[len(NegatifI)-i-1] = NegatifILoss[::-1][i]

    Rtab = np.concatenate((RtabPositif, RtabNegatif), axis = 0)
    Itab = np.concatenate((ItabPositif, ItabNegatif), axis = 0)

    Rtab[n] = RtabPositif[len(RtabPositif)-1]
    Itab[n] = ItabPositif[len(ItabPositif)-1]
    Rtab[0] = Rconstante
    Itab[0] = Iconstante

    tab2 = Rtab + Itab * 1j
    return tab2

def shift_negatif(tab, n): #utilisee dans la methode translation
    # constante = tab[0]
    #
    # tab = np.roll(tab,n)
    # for i in xrange(-n):
    #     tab[len(tab)-i-1] = 0
    # tab[0] = constante
    # return tab
    realTab = np.real(tab)
    imagTab = np.imag(tab)
    Rconstante = realTab[0]
    Iconstante = imagTab[0]

    RlossData = realTab[1 : -n+1]
    IlossData = imagTab[1 : -n+1]

    Rtab = np.roll(realTab,n)
    Itab = np.roll(imagTab,n)
    for i in xrange(-n):
        Rtab[len(tab)-i-1] = RlossData[i]
        Itab[len(tab)-i-1] = IlossData[i]
    Rtab[0] = Rconstante
    Itab[0] = Iconstante

    tab2 = Rtab + Itab * 1j
    return tab2
def translation (x,y, shift):
    matplotlib.rc('xtick', labelsize=15)
    matplotlib.rc('ytick', labelsize=15)
    f, axarr = plt.subplots(2, 3)
    fct = np.fft.fft(y)
    Re = abs(np.real(fct))
    Im = abs(np.imag(fct))
    mod = module_complex(fct)

    axarr[0, 0].plot(x[1:2000],Re[1:2000])
    axarr[0, 0].set_title('No shift Real', fontsize=15)
    #axarr[0, 0].set_xlabel('time (s)', fontsize=10)
    axarr[0, 0].set_ylabel('acceleration (m/$s^2$)', fontsize=15)
    axarr[0, 0].set_ylim((0, 250))

    axarr[0, 1].plot(x[1:2000],Im[1:2000])
    axarr[0, 1].set_title('No shift Imaginary', fontsize=15)
    #axarr[0, 1].set_xlabel('frequency (Hz)', fontsize=10)
    axarr[0, 1].set_ylabel('Amplitude', fontsize=10)
    axarr[0, 1].set_ylim((0, 250))



    axarr[0, 2].plot(x[1:2000], mod[1:2000])
    axarr[0, 2].set_title('No shift Modulus', fontsize=15)
    # axarr[0, 1].set_xlabel('frequency (Hz)', fontsize=10)
    axarr[0, 2].set_ylabel('Amplitude', fontsize=10)
    axarr[0, 2].set_ylim((0, 250))
    if shift > 0:
        fct = shift_positif2(fct, shift)
    else:
        fct = shift_negatif(fct, shift)

    Re = abs(np.real(fct))
    Im = abs(np.imag(fct))
    mod = module_complex(fct)

    axarr[1, 0].plot(x[1:2000],Re[1:2000])
    axarr[1, 0].set_title('2.5 Hz shift Real', fontsize=15)
    #axarr[0, 0].set_xlabel('time (s)', fontsize=10)
    axarr[1, 0].set_ylabel('acceleration (m/$s^2$)', fontsize=15)
    axarr[1, 0].set_ylim((0,250))

    axarr[1, 1].plot(x[1:2000],Im[1:2000])
    axarr[1, 1].set_title('2.5 Hz shift Imaginary', fontsize=15)
    #axarr[0, 1].set_xlabel('frequency (Hz)', fontsize=10)
    axarr[1, 1].set_ylabel('Amplitude', fontsize=10)
    axarr[1, 1].set_ylim((0, 250))

    axarr[1, 2].plot(x[1:2000], mod[1:2000])
    axarr[1, 2].set_title('2.5 Hz shift Modulus', fontsize=15)
    # axarr[0, 1].set_xlabel('frequency (Hz)', fontsize=10)
    axarr[1, 2].set_ylabel('Amplitude', fontsize=10)
    axarr[1, 2].set_ylim((0, 250))
    plt.show()

    return mod

def aleaGauss(sigma):
    #Methode de Box-Muller (voir wiki)
    U1 = random.random()
    U2 = random.random()
    return sigma*math.sqrt(-2*math.log(U1))*math.cos(2*math.pi*U2)
def bruit_gaussien(y, sigma):
    # matplotlib.rc('xtick', labelsize=15)
    # matplotlib.rc('ytick', labelsize=15)
    # f, axarr = plt.subplots(2, 2)
    # axarr[0, 0].plot(y[0:800])
    # axarr[0, 0].set_title('Temporel avant bruit de 30', fontsize=15)
    # #axarr[0, 0].set_xlabel('time (s)', fontsize=10)
    # axarr[0, 0].set_ylabel('acceleration (m/$s^2$)', fontsize=15)

    fct = np.fft.fft(y)
    real = np.real(fct)
    imag = np.imag(fct)

    # axarr[0, 1].plot(abs(real[1:]))
    # axarr[0, 1].set_title('Frequentiel avant bruit de 30', fontsize=15)
    # #axarr[0, 1].set_xlabel('frequency (Hz)', fontsize=10)
    # axarr[0, 1].set_ylabel('Amplitude', fontsize=10)



    for i in xrange(len(real)):
        real[i] = real[i] + aleaGauss(sigma)


    #
    # axarr[1, 1].plot(abs(real[1:]))
    # axarr[1, 1].set_title('Frequentiel apres bruit de 30', fontsize=15)
    # axarr[1, 1].set_xlabel('Frequency (Hz)', fontsize=15)
    # axarr[1, 1].set_ylabel('Amplitude', fontsize=15)



    fct1 = real + imag * 1j
    y = np.fft.ifft(fct1)

    # axarr[1, 0].plot(np.real(y)[0:800])
    # axarr[1, 0].set_title('Temporel apres bruit de 30', fontsize=15)
    # axarr[1, 0].set_xlabel('time (s)', fontsize=15)
    # axarr[1, 0].set_ylabel('acceleration (m/$s^2$)', fontsize=15)

    return np.real(y)
def laplace(k):
    return np.random.laplace(0,np.cbrt(k))
def caste_laplace(y,k):


    # matplotlib.rc('xtick', labelsize=15)
    # matplotlib.rc('ytick', labelsize=15)
    # f, axarr = plt.subplots(2, 2)
    # axarr[0, 0].plot(y[0:800])
    # axarr[0, 0].set_title('Temporel avant bruit de 30', fontsize=15)
    # #axarr[0, 0].set_xlabel('time (s)', fontsize=10)
    # axarr[0, 0].set_ylabel('acceleration (m/$s^2$)', fontsize=15)

    fct = np.fft.fft(y)
    real = np.real(fct)
    imag = np.imag(fct)

    # axarr[0, 1].plot(abs(real[1:]))
    # axarr[0, 1].set_title('Frequentiel avant bruit de 30', fontsize=15)
    # #axarr[0, 1].set_xlabel('frequency (Hz)', fontsize=10)
    # axarr[0, 1].set_ylabel('Amplitude', fontsize=10)

    n = int(len(real)/2)

    for i in xrange(len(real)):
        real[i] = real[i] + laplace(k)

    for i in xrange((n-k)*2):
        real[int(len(real)/2) - (n-k) + i ] = 0
        imag[int(len(imag) / 2) - (n - k) + i] = 0


    # axarr[1, 1].plot(abs(real[1:]))
    # axarr[1, 1].set_title('Frequentiel apres bruit de 30', fontsize=15)
    # axarr[1, 1].set_xlabel('Frequency (Hz)', fontsize=15)
    # axarr[1, 1].set_ylabel('Amplitude', fontsize=15)



    fct1 = real + imag * 1j
    y = np.fft.ifft(fct1)

    # axarr[1, 0].plot(np.real(y)[0:800])
    # axarr[1, 0].set_title('Temporel apres bruit de 30', fontsize=15)
    # axarr[1, 0].set_xlabel('time (s)', fontsize=15)
    # axarr[1, 0].set_ylabel('acceleration (m/$s^2$)', fontsize=15)

    return np.real(y)

def normalization_translation(data,new_maxi,new_mini):
    maxi = maximum(data)
    mini = minimum(data)
    newdata = []
    for i in xrange(len(data)):
        newdata.append((data[i] - mini)*(new_maxi - new_mini)/(maxi-mini) + new_mini)

    return newdata

def normalization_somme(data):
    dataCarre = []
    for i in xrange(len(data)):
        dataCarre.append(data[i]**2)
    rms = np.sqrt(np.mean(dataCarre))

    newdata = []
    for i in xrange(len(data)):
        newdata.append(data[i]/ rms)
    return newdata

def normalization_std(data, origine):
    stdData = standard_deviation(data)
    newdata = []
    for i in xrange(len(data)):
        newdata.append(data[i]*origine/stdData)
    return newdata


def normalization_iqr(data, origine):
    iqrData = iqr(data)
    newdata = []
    for i in xrange(len(data)):
        newdata.append(data[i]*origine/iqrData)
    return newdata


def normalization_mean(data, origine):
    meanData = mean(data)
    newdata = []
    for i in xrange(len(data)):
        newdata.append(data[i] - meanData + origine )
    return newdata
def fast_fourier_transform(data):
    return fftpack.rfft(data)

###################### CREATION DES DESCRIPTEURS ######################

def zero_crossing(tab):
    s = np.sign(tab)
    s[s == 0] = -1  # replace zeros with -1
    zero_crossings = np.where(np.diff(s))[0]
    return len(zero_crossings)

def mean(data):
    mean = 0
    for i in data:
        mean += i
    mean = mean/len(data)
    return mean

def maximum(data):
    return max(data)
def minimum(data):
    return min(data)

def variance(data):
    return np.var(data)
def mediane(data):
    return np.median(data)

def standard_deviation(data):
    return np.std(data)


def module_complex(complexvalue):
    return sqrt(complexvalue.real*complexvalue.real + complexvalue.imag*complexvalue.imag)

#Ecart inter quartile
def iqr(data):
    return scipy.stats.iqr(data)

def sma(X,Y,Z): #a faire
    return X+Y+Z

#Coefficient d'autocorrelation
# def acf(series):
#     n = len(series)
#     data = np.asarray(series)
#     mean = np.mean(data)
#     c0 = np.sum((data - mean) ** 2) / float(n)
#
#     def r(h):
#         acf_lag = ((data[:n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0
#         return round(acf_lag, 3)
#     x = np.arange(n) # avoiding lag 0 calculation
#     acf_coeffs = map(r, x)
#     return acf_coeffs

def autoregression (data, order):
    return arburg(data, order)

######## En frequentiel uniquement ########

def skewness(data):
    return scipy.stats.skew(data)
def kurtosis(data):
    return scipy.stats.kurtosis(data)

def spectral_energy(fftTab):
    abso = map(abs, fftTab)
    absoPow = []
    for i in abso: absoPow.append( i ** 2 )
    return sum(absoPow)/len(absoPow)

def maxFreqInd(data):
    return np.argmax(data)
def meanFreq(data): #comment  je definis les poids ???
    return data

def entropy(data, base=None):
    p_data = [abs(float(i)/(sum(data))) for i in data]  # calculates the probabilities
    ent = 0.0
    for i in p_data:
        # entropy.append(scipy.stats.entropy(i, base = base))  # input probabilities to get the entropy
        ent -= i * log(i+0.0000000000001, None)
    return ent

def entropy2(data):
    probs = [np.mean(float(data == c)) for c in set(data)]

    return np.sum(-p * np.log2(p) for p in probs)

def entropy3(data):
    scipy.stats.entropy(data)
    return scipy.stats.entropy(data)
#rien a voir (pour biosignalplux)
#http://www.silabs.com/documents/login/software/BLED112-Signed-Win-Drv.zip
