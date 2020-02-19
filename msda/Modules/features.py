import tqdm
import torch
import pandas as pd
import numpy as np
import scipy.signal as signal
import time
import matplotlib.pyplot as plt
import math
from Modules.PreProcessing import *
import scipy.fftpack as fftpack
import scipy
from operator import itemgetter
import csv
import os

def features_creation(adressSave, adressTrain, adressTest, epoch)
	print("Epoch : " + str(epoch))
	
	if not os.path.exists(adressSave):
		os.mkdir(adressSave, mode = 0o777)

	train = pd.read_csv(adressTrain)
	test = pd.read_csv(adressTest)


	acts_ids = test.loc[:,'act':'id']
	X1test = []
	Y1test = []
	Z1test = []
	X2test = []
	Y2test = []
	Z2test = []
	gender_test = []
	act_test = []
	for k in range(24):
		temp1 = acts_ids[acts_ids.loc[:,"id"] == k]
		for l in range(4):
			curseur = temp1[temp1.loc[:,"act"] ==l]
			myTest = test.loc[curseur.index,:]
			myTest = slidingWindow(myTest, 128, 64)
			myTest.pop(len(myTest)-1)
			
			for m in myTest[:][:]:
				X1test.append(m.loc[:,"userAcceleration.x"].values)
				Y1test.append(m.loc[:,"userAcceleration.y"].values)
				Z1test.append(m.loc[:,"userAcceleration.z"].values)
				X2test.append(m.loc[:,"rotationRate.x"].values)
				Y2test.append(m.loc[:,"rotationRate.y"].values)
				Z2test.append(m.loc[:,"rotationRate.z"].values)
				gender_test.append(m.loc[:,"gender"].values[0])
				act_test.append(m.loc[:,"act"].values[0])

	#with open('yTest_gender.csv', 'w') as file:
	#	writer = csv.writer(file)
	#	for row in zip(gender):
	#		writer.writerow(row)
	#with open('yTest_act.csv', 'w') as file:
	#	writer = csv.writer(file)
	#	for row in zip(act):
	#		writer.writerow(row)
	X1testF = []
	Y1testF = []
	Z1testF = []
	X2testF = []
	Y2testF = []
	Z2testF = []

	for j in range(len(X1test)):
		X1testF.append(abs(scipy.fftpack.fft(X1test[j])))
		Y1testF.append(abs(scipy.fftpack.fft(Y1test[j])))
		Z1testF.append(abs(scipy.fftpack.fft(Z1test[j])))

		X2testF.append(abs(scipy.fftpack.fft(X2test[j])))
		Y2testF.append(abs(scipy.fftpack.fft(Y2test[j])))
		Z2testF.append(abs(scipy.fftpack.fft(Z2test[j])))



	#############################################################################################
	X1test_mean = []
	Y1test_mean = []
	Z1test_mean = []
	X2test_mean = []
	Y2test_mean = []
	Z2test_mean = []

	for i in range(len(X1test)):
		X1test_mean.append(np.mean(X1test[i]))
		Y1test_mean.append(np.mean(Y1test[i]))
		Z1test_mean.append(np.mean(Z1test[i]))

		X2test_mean.append(np.mean(X2test[i]))
		Y2test_mean.append(np.mean(Y2test[i]))
		Z2test_mean.append(np.mean(Z2test[i]))


	X1test_min = []
	Y1test_min = []
	Z1test_min = []
	X2test_min = []
	Y2test_min = []
	Z2test_min = []

	for i in range(len(X1test)):
		X1test_min.append(np.min(X1test[i]))
		Y1test_min.append(np.min(Y1test[i]))
		Z1test_min.append(np.min(Z1test[i]))

		X2test_min.append(np.min(X2test[i]))
		Y2test_min.append(np.min(Y2test[i]))
		Z2test_min.append(np.min(Z2test[i]))

	X1test_max = []
	Y1test_max = []
	Z1test_max = []
	X2test_max = []
	Y2test_max = []
	Z2test_max = []

	for i in range(len(X1test)):
		X1test_max.append(np.max(X1test[i]))
		Y1test_max.append(np.max(Y1test[i]))
		Z1test_max.append(np.max(Z1test[i]))

		X2test_max.append(np.max(X2test[i]))
		Y2test_max.append(np.max(Y2test[i]))
		Z2test_max.append(np.max(Z2test[i]))

	X1test_std = []
	Y1test_std = []
	Z1test_std = []
	X2test_std = []
	Y2test_std = []
	Z2test_std = []

	for i in range(len(X1test)):
		X1test_std.append(np.std(X1test[i]))
		Y1test_std.append(np.std(Y1test[i]))
		Z1test_std.append(np.std(Z1test[i]))

		X2test_std.append(np.std(X2test[i]))
		Y2test_std.append(np.std(Y2test[i]))
		Z2test_std.append(np.std(Z2test[i]))


	X1test_iqr = []
	Y1test_iqr = []
	Z1test_iqr = []
	X2test_iqr = []
	Y2test_iqr = []
	Z2test_iqr = []

	for i in range(len(X1test)):
		X1test_iqr.append(scipy.stats.iqr(X1test[i]))
		Y1test_iqr.append(scipy.stats.iqr(Y1test[i]))
		Z1test_iqr.append(scipy.stats.iqr(Z1test[i]))

		X2test_iqr.append(scipy.stats.iqr(X2test[i]))
		Y2test_iqr.append(scipy.stats.iqr(Y2test[i]))
		Z2test_iqr.append(scipy.stats.iqr(Z2test[i]))

	X1test_var = []
	Y1test_var = []
	Z1test_var = []
	X2test_var = []
	Y2test_var = []
	Z2test_var = []

	for i in range(len(X1test)):
		X1test_var.append(np.var(X1test[i]))
		Y1test_var.append(np.var(Y1test[i]))
		Z1test_var.append(np.var(Z1test[i]))

		X2test_var.append(np.var(X2test[i]))
		Y2test_var.append(np.var(Y2test[i]))
		Z2test_var.append(np.var(Z2test[i]))


	X1test_skew = []
	Y1test_skew = []
	Z1test_skew = []
	X2test_skew = []
	Y2test_skew = []
	Z2test_skew = []

	for i in range(len(X1test)):
		X1test_skew.append(skewness(X1test[i]))
		Y1test_skew.append(skewness(Y1test[i]))
		Z1test_skew.append(skewness(Z1test[i]))

		X2test_skew.append(skewness(X2test[i]))
		Y2test_skew.append(skewness(Y2test[i]))
		Z2test_skew.append(skewness(Z2test[i]))

	X1test_kurt = []
	Y1test_kurt = []
	Z1test_kurt = []
	X2test_kurt = []
	Y2test_kurt = []
	Z2test_kurt = []

	for i in range(len(X1test)):
		X1test_kurt.append(kurtosis(X1test[i]))
		Y1test_kurt.append(kurtosis(Y1test[i]))
		Z1test_kurt.append(kurtosis(Z1test[i]))

		X2test_kurt.append(kurtosis(X2test[i]))
		Y2test_kurt.append(kurtosis(Y2test[i]))
		Z2test_kurt.append(kurtosis(Z2test[i]))


	X1test_med = []
	Y1test_med = []
	Z1test_med = []
	X2test_med = []
	Y2test_med = []
	Z2test_med = []

	for i in range(len(X1test)):
		X1test_med.append(np.median(X1test[i]))
		Y1test_med.append(np.median(Y1test[i]))
		Z1test_med.append(np.median(Z1test[i]))

		X2test_med.append(np.median(X2test[i]))
		Y2test_med.append(np.median(Y2test[i]))
		Z2test_med.append(np.median(Z2test[i]))
	###################################################################################################################"
	#############################################################################################
	X1testF_mean = []
	Y1testF_mean = []
	Z1testF_mean = []
	X2testF_mean = []
	Y2testF_mean = []
	Z2testF_mean = []

	for i in range(len(X1test)):
		X1testF_mean.append(np.mean(X1testF[i]))
		Y1testF_mean.append(np.mean(Y1testF[i]))
		Z1testF_mean.append(np.mean(Z1testF[i]))

		X2testF_mean.append(np.mean(X2testF[i]))
		Y2testF_mean.append(np.mean(Y2testF[i]))
		Z2testF_mean.append(np.mean(Z2testF[i]))


	X1testF_min = []
	Y1testF_min = []
	Z1testF_min = []
	X2testF_min = []
	Y2testF_min = []
	Z2testF_min = []

	for i in range(len(X1test)):
		X1testF_min.append(np.min(X1testF[i]))
		Y1testF_min.append(np.min(Y1testF[i]))
		Z1testF_min.append(np.min(Z1testF[i]))

		X2testF_min.append(np.min(X2testF[i]))
		Y2testF_min.append(np.min(Y2testF[i]))
		Z2testF_min.append(np.min(Z2testF[i]))

	X1testF_max = []
	Y1testF_max = []
	Z1testF_max = []
	X2testF_max = []
	Y2testF_max = []
	Z2testF_max = []

	for i in range(len(X1test)):
		X1testF_max.append(np.max(X1testF[i]))
		Y1testF_max.append(np.max(Y1testF[i]))
		Z1testF_max.append(np.max(Z1testF[i]))

		X2testF_max.append(np.max(X2testF[i]))
		Y2testF_max.append(np.max(Y2testF[i]))
		Z2testF_max.append(np.max(Z2testF[i]))

	X1testF_std = []
	Y1testF_std = []
	Z1testF_std = []
	X2testF_std = []
	Y2testF_std = []
	Z2testF_std = []

	for i in range(len(X1test)):
		X1testF_std.append(np.std(X1testF[i]))
		Y1testF_std.append(np.std(Y1testF[i]))
		Z1testF_std.append(np.std(Z1testF[i]))

		X2testF_std.append(np.std(X2testF[i]))
		Y2testF_std.append(np.std(Y2testF[i]))
		Z2testF_std.append(np.std(Z2testF[i]))


	X1testF_iqr = []
	Y1testF_iqr = []
	Z1testF_iqr = []
	X2testF_iqr = []
	Y2testF_iqr = []
	Z2testF_iqr = []

	for i in range(len(X1test)):
		X1testF_iqr.append(scipy.stats.iqr(X1testF[i]))
		Y1testF_iqr.append(scipy.stats.iqr(Y1testF[i]))
		Z1testF_iqr.append(scipy.stats.iqr(Z1testF[i]))

		X2testF_iqr.append(scipy.stats.iqr(X2testF[i]))
		Y2testF_iqr.append(scipy.stats.iqr(Y2testF[i]))
		Z2testF_iqr.append(scipy.stats.iqr(Z2testF[i]))

	X1testF_var = []
	Y1testF_var = []
	Z1testF_var = []
	X2testF_var = []
	Y2testF_var = []
	Z2testF_var = []

	for i in range(len(X1test)):
		X1testF_var.append(np.var(X1testF[i]))
		Y1testF_var.append(np.var(Y1testF[i]))
		Z1testF_var.append(np.var(Z1testF[i]))

		X2testF_var.append(np.var(X2testF[i]))
		Y2testF_var.append(np.var(Y2testF[i]))
		Z2testF_var.append(np.var(Z2testF[i]))


	X1testF_skew = []
	Y1testF_skew = []
	Z1testF_skew = []
	X2testF_skew = []
	Y2testF_skew = []
	Z2testF_skew = []

	for i in range(len(X1test)):
		X1testF_skew.append(skewness(X1testF[i]))
		Y1testF_skew.append(skewness(Y1testF[i]))
		Z1testF_skew.append(skewness(Z1testF[i]))

		X2testF_skew.append(skewness(X2testF[i]))
		Y2testF_skew.append(skewness(Y2testF[i]))
		Z2testF_skew.append(skewness(Z2testF[i]))

	X1testF_kurt = []
	Y1testF_kurt = []
	Z1testF_kurt = []
	X2testF_kurt = []
	Y2testF_kurt = []
	Z2testF_kurt = []

	for i in range(len(X1test)):
		X1testF_kurt.append(kurtosis(X1testF[i]))
		Y1testF_kurt.append(kurtosis(Y1testF[i]))
		Z1testF_kurt.append(kurtosis(Z1testF[i]))

		X2testF_kurt.append(kurtosis(X2testF[i]))
		Y2testF_kurt.append(kurtosis(Y2testF[i]))
		Z2testF_kurt.append(kurtosis(Z2testF[i]))


	X1testF_med = []
	Y1testF_med = []
	Z1testF_med = []
	X2testF_med = []
	Y2testF_med = []
	Z2testF_med = []

	for i in range(len(X1test)):
		X1testF_med.append(np.median(X1testF[i]))
		Y1testF_med.append(np.median(Y1testF[i]))
		Z1testF_med.append(np.median(Z1testF[i]))

		X2testF_med.append(np.median(X2testF[i]))
		Y2testF_med.append(np.median(Y2testF[i]))
		Z2testF_med.append(np.median(Z2testF[i]))
	###################################################################################################################

	###################################################################################################################

	###################################################################################################################

	acts_ids = train.loc[:,'act':'id']


	X1train = []
	Y1train = []
	Z1train = []
	X2train = []
	Y2train = []
	Z2train = []
	gender_train = []
	act_train = []
	for k in range(24):
		temp1 = acts_ids[acts_ids.loc[:,"id"] == k]
		for l in range(4):
			curseur = temp1[temp1.loc[:,"act"] ==l]
			myTrain = train.loc[curseur.index,:]
			myTrain = slidingWindow(myTrain, 128, 64)
			myTrain.pop(len(myTrain)-1)
			
			for m in myTrain[:][:]:
				X1train.append(m.loc[:,"userAcceleration.x"].values)
				Y1train.append(m.loc[:,"userAcceleration.y"].values)
				Z1train.append(m.loc[:,"userAcceleration.z"].values)
				X2train.append(m.loc[:,"rotationRate.x"].values)
				Y2train.append(m.loc[:,"rotationRate.y"].values)
				Z2train.append(m.loc[:,"rotationRate.z"].values)
				gender_train.append(m.loc[:,"gender"].values[0])
				act_train.append(m.loc[:,"act"].values[0])

	#with open('yTrain_gender.csv', 'w') as file:
	#	writer = csv.writer(file)
	#	for row in zip(gender):
	#		writer.writerow(row)
	#with open('yTrain_act.csv', 'w') as file:
	#	writer = csv.writer(file)
	#	for row in zip(act):
	#		writer.writerow(row)

	X1trainF = []
	Y1trainF = []
	Z1trainF = []
	X2trainF = []
	Y2trainF = []
	Z2trainF = []

	for j in range(len(X1train)):
		X1trainF.append(abs(scipy.fftpack.fft(X1train[j])))
		Y1trainF.append(abs(scipy.fftpack.fft(Y1train[j])))
		Z1trainF.append(abs(scipy.fftpack.fft(Z1train[j])))

		X2trainF.append(abs(scipy.fftpack.fft(X2train[j])))
		Y2trainF.append(abs(scipy.fftpack.fft(Y2train[j])))
		Z2trainF.append(abs(scipy.fftpack.fft(Z2train[j])))



	#############################################################################################
	X1train_mean = []
	Y1train_mean = []
	Z1train_mean = []
	X2train_mean = []
	Y2train_mean = []
	Z2train_mean = []

	for i in range(len(X1train)):
		X1train_mean.append(np.mean(X1train[i]))
		Y1train_mean.append(np.mean(Y1train[i]))
		Z1train_mean.append(np.mean(Z1train[i]))

		X2train_mean.append(np.mean(X2train[i]))
		Y2train_mean.append(np.mean(Y2train[i]))
		Z2train_mean.append(np.mean(Z2train[i]))


	X1train_min = []
	Y1train_min = []
	Z1train_min = []
	X2train_min = []
	Y2train_min = []
	Z2train_min = []

	for i in range(len(X1train)):
		X1train_min.append(np.min(X1train[i]))
		Y1train_min.append(np.min(Y1train[i]))
		Z1train_min.append(np.min(Z1train[i]))

		X2train_min.append(np.min(X2train[i]))
		Y2train_min.append(np.min(Y2train[i]))
		Z2train_min.append(np.min(Z2train[i]))

	X1train_max = []
	Y1train_max = []
	Z1train_max = []
	X2train_max = []
	Y2train_max = []
	Z2train_max = []

	for i in range(len(X1train)):
		X1train_max.append(np.max(X1train[i]))
		Y1train_max.append(np.max(Y1train[i]))
		Z1train_max.append(np.max(Z1train[i]))

		X2train_max.append(np.max(X2train[i]))
		Y2train_max.append(np.max(Y2train[i]))
		Z2train_max.append(np.max(Z2train[i]))

	X1train_std = []
	Y1train_std = []
	Z1train_std = []
	X2train_std = []
	Y2train_std = []
	Z2train_std = []

	for i in range(len(X1train)):
		X1train_std.append(np.std(X1train[i]))
		Y1train_std.append(np.std(Y1train[i]))
		Z1train_std.append(np.std(Z1train[i]))

		X2train_std.append(np.std(X2train[i]))
		Y2train_std.append(np.std(Y2train[i]))
		Z2train_std.append(np.std(Z2train[i]))


	X1train_iqr = []
	Y1train_iqr = []
	Z1train_iqr = []
	X2train_iqr = []
	Y2train_iqr = []
	Z2train_iqr = []

	for i in range(len(X1train)):
		X1train_iqr.append(scipy.stats.iqr(X1train[i]))
		Y1train_iqr.append(scipy.stats.iqr(Y1train[i]))
		Z1train_iqr.append(scipy.stats.iqr(Z1train[i]))

		X2train_iqr.append(scipy.stats.iqr(X2train[i]))
		Y2train_iqr.append(scipy.stats.iqr(Y2train[i]))
		Z2train_iqr.append(scipy.stats.iqr(Z2train[i]))

	X1train_var = []
	Y1train_var = []
	Z1train_var = []
	X2train_var = []
	Y2train_var = []
	Z2train_var = []

	for i in range(len(X1train)):
		X1train_var.append(np.var(X1train[i]))
		Y1train_var.append(np.var(Y1train[i]))
		Z1train_var.append(np.var(Z1train[i]))

		X2train_var.append(np.var(X2train[i]))
		Y2train_var.append(np.var(Y2train[i]))
		Z2train_var.append(np.var(Z2train[i]))


	X1train_skew = []
	Y1train_skew = []
	Z1train_skew = []
	X2train_skew = []
	Y2train_skew = []
	Z2train_skew = []

	for i in range(len(X1train)):
		X1train_skew.append(skewness(X1train[i]))
		Y1train_skew.append(skewness(Y1train[i]))
		Z1train_skew.append(skewness(Z1train[i]))

		X2train_skew.append(skewness(X2train[i]))
		Y2train_skew.append(skewness(Y2train[i]))
		Z2train_skew.append(skewness(Z2train[i]))

	X1train_kurt = []
	Y1train_kurt = []
	Z1train_kurt = []
	X2train_kurt = []
	Y2train_kurt = []
	Z2train_kurt = []

	for i in range(len(X1train)):
		X1train_kurt.append(kurtosis(X1train[i]))
		Y1train_kurt.append(kurtosis(Y1train[i]))
		Z1train_kurt.append(kurtosis(Z1train[i]))

		X2train_kurt.append(kurtosis(X2train[i]))
		Y2train_kurt.append(kurtosis(Y2train[i]))
		Z2train_kurt.append(kurtosis(Z2train[i]))



	X1train_med = []
	Y1train_med = []
	Z1train_med = []
	X2train_med = []
	Y2train_med = []
	Z2train_med = []

	for i in range(len(X1train)):
		X1train_med.append(np.median(X1train[i]))
		Y1train_med.append(np.median(Y1train[i]))
		Z1train_med.append(np.median(Z1train[i]))

		X2train_med.append(np.median(X2train[i]))
		Y2train_med.append(np.median(Y2train[i]))
		Z2train_med.append(np.median(Z2train[i]))
	###################################################################################################################"
	#############################################################################################
	X1trainF_mean = []
	Y1trainF_mean = []
	Z1trainF_mean = []
	X2trainF_mean = []
	Y2trainF_mean = []
	Z2trainF_mean = []

	for i in range(len(X1train)):
		X1trainF_mean.append(np.mean(X1trainF[i]))
		Y1trainF_mean.append(np.mean(Y1trainF[i]))
		Z1trainF_mean.append(np.mean(Z1trainF[i]))

		X2trainF_mean.append(np.mean(X2trainF[i]))
		Y2trainF_mean.append(np.mean(Y2trainF[i]))
		Z2trainF_mean.append(np.mean(Z2trainF[i]))


	X1trainF_min = []
	Y1trainF_min = []
	Z1trainF_min = []
	X2trainF_min = []
	Y2trainF_min = []
	Z2trainF_min = []

	for i in range(len(X1train)):
		X1trainF_min.append(np.min(X1trainF[i]))
		Y1trainF_min.append(np.min(Y1trainF[i]))
		Z1trainF_min.append(np.min(Z1trainF[i]))

		X2trainF_min.append(np.min(X2trainF[i]))
		Y2trainF_min.append(np.min(Y2trainF[i]))
		Z2trainF_min.append(np.min(Z2trainF[i]))

	X1trainF_max = []
	Y1trainF_max = []
	Z1trainF_max = []
	X2trainF_max = []
	Y2trainF_max = []
	Z2trainF_max = []

	for i in range(len(X1train)):
		X1trainF_max.append(np.max(X1trainF[i]))
		Y1trainF_max.append(np.max(Y1trainF[i]))
		Z1trainF_max.append(np.max(Z1trainF[i]))

		X2trainF_max.append(np.max(X2trainF[i]))
		Y2trainF_max.append(np.max(Y2trainF[i]))
		Z2trainF_max.append(np.max(Z2trainF[i]))

	X1trainF_std = []
	Y1trainF_std = []
	Z1trainF_std = []
	X2trainF_std = []
	Y2trainF_std = []
	Z2trainF_std = []

	for i in range(len(X1train)):
		X1trainF_std.append(np.std(X1trainF[i]))
		Y1trainF_std.append(np.std(Y1trainF[i]))
		Z1trainF_std.append(np.std(Z1trainF[i]))

		X2trainF_std.append(np.std(X2trainF[i]))
		Y2trainF_std.append(np.std(Y2trainF[i]))
		Z2trainF_std.append(np.std(Z2trainF[i]))


	X1trainF_iqr = []
	Y1trainF_iqr = []
	Z1trainF_iqr = []
	X2trainF_iqr = []
	Y2trainF_iqr = []
	Z2trainF_iqr = []

	for i in range(len(X1train)):
		X1trainF_iqr.append(scipy.stats.iqr(X1trainF[i]))
		Y1trainF_iqr.append(scipy.stats.iqr(Y1trainF[i]))
		Z1trainF_iqr.append(scipy.stats.iqr(Z1trainF[i]))

		X2trainF_iqr.append(scipy.stats.iqr(X2trainF[i]))
		Y2trainF_iqr.append(scipy.stats.iqr(Y2trainF[i]))
		Z2trainF_iqr.append(scipy.stats.iqr(Z2trainF[i]))

	X1trainF_var = []
	Y1trainF_var = []
	Z1trainF_var = []
	X2trainF_var = []
	Y2trainF_var = []
	Z2trainF_var = []

	for i in range(len(X1train)):
		X1trainF_var.append(np.var(X1trainF[i]))
		Y1trainF_var.append(np.var(Y1trainF[i]))
		Z1trainF_var.append(np.var(Z1trainF[i]))

		X2trainF_var.append(np.var(X2trainF[i]))
		Y2trainF_var.append(np.var(Y2trainF[i]))
		Z2trainF_var.append(np.var(Z2trainF[i]))


	X1trainF_skew = []
	Y1trainF_skew = []
	Z1trainF_skew = []
	X2trainF_skew = []
	Y2trainF_skew = []
	Z2trainF_skew = []

	for i in range(len(X1train)):
		X1trainF_skew.append(skewness(X1trainF[i]))
		Y1trainF_skew.append(skewness(Y1trainF[i]))
		Z1trainF_skew.append(skewness(Z1trainF[i]))

		X2trainF_skew.append(skewness(X2trainF[i]))
		Y2trainF_skew.append(skewness(Y2trainF[i]))
		Z2trainF_skew.append(skewness(Z2trainF[i]))

	X1trainF_kurt = []
	Y1trainF_kurt = []
	Z1trainF_kurt = []
	X2trainF_kurt = []
	Y2trainF_kurt = []
	Z2trainF_kurt = []

	for i in range(len(X1train)):
		X1trainF_kurt.append(kurtosis(X1trainF[i]))
		Y1trainF_kurt.append(kurtosis(Y1trainF[i]))
		Z1trainF_kurt.append(kurtosis(Z1trainF[i]))

		X2trainF_kurt.append(kurtosis(X2trainF[i]))
		Y2trainF_kurt.append(kurtosis(Y2trainF[i]))
		Z2trainF_kurt.append(kurtosis(Z2trainF[i]))


	X1trainF_med = []
	Y1trainF_med = []
	Z1trainF_med = []
	X2trainF_med = []
	Y2trainF_med = []
	Z2trainF_med = []

	for i in range(len(X1train)):
		X1trainF_med.append(np.median(X1trainF[i]))
		Y1trainF_med.append(np.median(Y1trainF[i]))
		Z1trainF_med.append(np.median(Z1trainF[i]))

		X2trainF_med.append(np.median(X2trainF[i]))
		Y2trainF_med.append(np.median(Y2trainF[i]))
		Z2trainF_med.append(np.median(Z2trainF[i]))
	###################################################################################################################

	import csv

	names  = ["X1test_mean","Y1test_mean","Z1test_mean","X2test_mean","Y2test_mean","Z2test_mean",
		"X1test_min","Y1test_min","Z1test_min","X2test_min","Y2test_min","Z2test_min",
		"X1test_max","Y1test_max","Z1test_max","X2test_max","Y2test_max","Z2test_max",
		"X1test_std","Y1test_std","Z1test_std","X2test_std","Y2test_std","Z2test_std",
		"X1test_var","Y1test_var","Z1test_var","X2test_var","Y2test_var","Z2test_var",
		"X1test_iqr","Y1test_iqr","Z1test_iqr","X2test_iqr","Y2test_iqr","Z2test_iqr",
		"X1test_med","Y1test_med","Z1test_med","X2test_med","Y2test_med","Z2test_med",
		"X1test_skew","Y1test_skew","Z1test_skew","X2test_skew","Y2test_skew","Z2test_skew",
		"X1test_kurt","Y1test_kurt","Z1test_kurt","X2test_kurt","Y2test_kurt","Z2test_kurt",

		"X1testF_mean","Y1testF_mean","Z1testF_mean","X2testF_mean","Y2testF_mean","Z2testF_mean",
		"X1testF_min","Y1testF_min","Z1testF_min","X2testF_min","Y2testF_min","Z2testF_min",
		"X1testF_max","Y1testF_max","Z1testF_max","X2testF_max","Y2testF_max","Z2testF_max",
		"X1testF_std","Y1testF_std","Z1testF_std","X2testF_std","Y2testF_std","Z2testF_std",
		"X1testF_var","Y1testF_var","Z1testF_var","X2testF_var","Y2testF_var","Z2testF_var",
		"X1testF_iqr","Y1testF_iqr","Z1testF_iqr","X2testF_iqr","Y2testF_iqr","Z2testF_iqr",
		"X1testF_med","Y1testF_med","Z1testF_med","X2testF_med","Y2testF_med","Z2testF_med",
		"X1testF_skew","Y1testF_skew","Z1testF_skew","X2testF_skew","Y2testF_skew","Z2testF_skew",
		"X1testF_kurt","Y1testF_kurt","Z1testF_kurt","X2testF_kurt","Y2testF_kurt","Z2testF_kurt", "gender", "act"]


	data_features = zip(X1test_mean,Y1test_mean,Z1test_mean,X2test_mean,Y2test_mean,Z2test_mean,
		X1test_min,Y1test_min,Z1test_min,X2test_min,Y2test_min,Z2test_min,
		X1test_max,Y1test_max,Z1test_max,X2test_max,Y2test_max,Z2test_max,
		X1test_std,Y1test_std,Z1test_std,X2test_std,Y2test_std,Z2test_std,
		X1test_var,Y1test_var,Z1test_var,X2test_var,Y2test_var,Z2test_var,
		X1test_iqr,Y1test_iqr,Z1test_iqr,X2test_iqr,Y2test_iqr,Z2test_iqr,
		X1test_med,Y1test_med,Z1test_med,X2test_med,Y2test_med,Z2test_med,
		X1test_skew,Y1test_skew,Z1test_skew,X2test_skew,Y2test_skew,Z2test_skew,
		X1test_kurt,Y1test_kurt,Z1test_kurt,X2test_kurt,Y2test_kurt,Z2test_kurt,

		X1testF_mean,Y1testF_mean,Z1testF_mean,X2testF_mean,Y2testF_mean,Z2testF_mean,
		X1testF_min,Y1testF_min,Z1testF_min,X2testF_min,Y2testF_min,Z2testF_min,
		X1testF_max,Y1testF_max,Z1testF_max,X2testF_max,Y2testF_max,Z2testF_max,
		X1testF_std,Y1testF_std,Z1testF_std,X2testF_std,Y2testF_std,Z2testF_std,
		X1testF_var,Y1testF_var,Z1testF_var,X2testF_var,Y2testF_var,Z2testF_var,
		X1testF_iqr,Y1testF_iqr,Z1testF_iqr,X2testF_iqr,Y2testF_iqr,Z2testF_iqr,
		X1testF_med,Y1testF_med,Z1testF_med,X2testF_med,Y2testF_med,Z2testF_med,
		X1testF_skew,Y1testF_skew,Z1testF_skew,X2testF_skew,Y2testF_skew,Z2testF_skew,
		X1testF_kurt,Y1testF_kurt,Z1testF_kurt,X2testF_kurt,Y2testF_kurt,Z2testF_kurt,gender_test, act_test)

	with open(adressSave + '/features_Test_'+ str(epoch) +'.csv', 'w') as file:
		writer = csv.writer(file)
		writer.writerow(names)
		for row in data_features:
			writer.writerow(row)


	#with open('namesTest.csv', 'w') as file:
	#	writer = csv.writer(file)
	#	writer.writerow(names)


	names  = ["X1train_mean","Y1train_mean","Z1train_mean","X2train_mean","Y2train_mean","Z2train_mean",
		"X1train_min","Y1train_min","Z1train_min","X2train_min","Y2train_min","Z2train_min",
		"X1train_max","Y1train_max","Z1train_max","X2train_max","Y2train_max","Z2train_max",
		"X1train_std","Y1train_std","Z1train_std","X2train_std","Y2train_std","Z2train_std",
		"X1train_var","Y1train_var","Z1train_var","X2train_var","Y2train_var","Z2train_var",
		"X1train_iqr","Y1train_iqr","Z1train_iqr","X2train_iqr","Y2train_iqr","Z2train_iqr",
		"X1train_med","Y1train_med","Z1train_med","X2train_med","Y2train_med","Z2train_med",
		"X1train_skew","Y1train_skew","Z1train_skew","X2train_skew","Y2train_skew","Z2train_skew",
		"X1train_kurt","Y1train_kurt","Z1train_kurt","X2train_kurt","Y2train_kurt","Z2train_kurt",

		"X1trainF_mean","Y1trainF_mean","Z1trainF_mean","X2trainF_mean","Y2trainF_mean","Z2trainF_mean",
		"X1trainF_min","Y1trainF_min","Z1trainF_min","X2trainF_min","Y2trainF_min","Z2trainF_min",
		"X1trainF_max","Y1trainF_max","Z1trainF_max","X2trainF_max","Y2trainF_max","Z2trainF_max",
		"X1trainF_std","Y1trainF_std","Z1trainF_std","X2trainF_std","Y2trainF_std","Z2trainF_std",
		"X1trainF_var","Y1trainF_var","Z1trainF_var","X2trainF_var","Y2trainF_var","Z2trainF_var",
		"X1trainF_iqr","Y1trainF_iqr","Z1trainF_iqr","X2trainF_iqr","Y2trainF_iqr","Z2trainF_iqr",
		"X1trainF_med","Y1trainF_med","Z1trainF_med","X2trainF_med","Y2trainF_med","Z2trainF_med",
		"X1trainF_skew","Y1trainF_skew","Z1trainF_skew","X2trainF_skew","Y2trainF_skew","Z2trainF_skew",
		"X1trainF_kurt","Y1trainF_kurt","Z1trainF_kurt","X2trainF_kurt","Y2trainF_kurt","Z2trainF_kurt", "gender", "act"]

	data_features = zip(X1train_mean,Y1train_mean,Z1train_mean,X2train_mean,Y2train_mean,Z2train_mean,
		X1train_min,Y1train_min,Z1train_min,X2train_min,Y2train_min,Z2train_min,
		X1train_max,Y1train_max,Z1train_max,X2train_max,Y2train_max,Z2train_max,
		X1train_std,Y1train_std,Z1train_std,X2train_std,Y2train_std,Z2train_std,
		X1train_var,Y1train_var,Z1train_var,X2train_var,Y2train_var,Z2train_var,
		X1train_iqr,Y1train_iqr,Z1train_iqr,X2train_iqr,Y2train_iqr,Z2train_iqr,
		X1train_med,Y1train_med,Z1train_med,X2train_med,Y2train_med,Z2train_med,
		X1train_skew,Y1train_skew,Z1train_skew,X2train_skew,Y2train_skew,Z2train_skew,
		X1train_kurt,Y1train_kurt,Z1train_kurt,X2train_kurt,Y2train_kurt,Z2train_kurt,

		X1trainF_mean,Y1trainF_mean,Z1trainF_mean,X2trainF_mean,Y2trainF_mean,Z2trainF_mean,
		X1trainF_min,Y1trainF_min,Z1trainF_min,X2trainF_min,Y2trainF_min,Z2trainF_min,
		X1trainF_max,Y1trainF_max,Z1trainF_max,X2trainF_max,Y2trainF_max,Z2trainF_max,
		X1trainF_std,Y1trainF_std,Z1trainF_std,X2trainF_std,Y2trainF_std,Z2trainF_std,
		X1trainF_var,Y1trainF_var,Z1trainF_var,X2trainF_var,Y2trainF_var,Z2trainF_var,
		X1trainF_iqr,Y1trainF_iqr,Z1trainF_iqr,X2trainF_iqr,Y2trainF_iqr,Z2trainF_iqr,
		X1trainF_med,Y1trainF_med,Z1trainF_med,X2trainF_med,Y2trainF_med,Z2trainF_med,
		X1trainF_skew,Y1trainF_skew,Z1trainF_skew,X2trainF_skew,Y2trainF_skew,Z2trainF_skew,
		X1trainF_kurt,Y1trainF_kurt,Z1trainF_kurt,X2trainF_kurt,Y2trainF_kurt,Z2trainF_kurt, gender_train, act_train)

	with open(adressSave +'/features_Train_'+ str(epoch) +'.csv', 'w') as file:
		writer = csv.writer(file)
		writer.writerow(names)
		for row in data_features:
			writer.writerow(row)

