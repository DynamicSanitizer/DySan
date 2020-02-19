import tqdm
import torch
from Parameters import Parameters as P
from Modules import Setup as S
from Modules import Models as M
from Modules import Results as R
from Modules import Metrics as Me
from Modules import Datasets as D
from Modules import Datasets2 as D2
from Modules import CustomLosses as Cl
from Modules import features
import PreProcessing
import pandas as pd
from PreProcessing import *
import numpy as np
import scipy.signal as signal
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import scipy.fftpack as fftpack
import scipy
from operator import itemgetter
import csv
import os
from sklearn.metrics import accuracy_score
from scipy.spatial import distance as Di
from sklearn import ensemble, neural_network, svm, gaussian_process, linear_model, tree
from torch.utils import data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")
torch.cuda.empty_cache()

originaleTrain = "../data/trainTrial.csv"
originaleTest = "../data/testTrial.csv"
adressSan = P.ExperimentBaseDir + P.SetName + "/Exp_" + P.ExpNumber +  "/SubExp_1/Generated/"
adressSave = P.ExperimentBaseDir + P.SetName + "/Exp_" + P.ExpNumber +  "/features"
epochs = [1,25,50,75,100,125,150,175,200,225,250,275,300]

seed = 42
batch_size = 256

tc = [
	ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed),
	neural_network.MLPClassifier(random_state=seed),
	tree.DecisionTreeClassifier(),
	ensemble.RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed),
	linear_model.LogisticRegression(class_weight='balanced', solver='liblinear')
	]
numberClass = len(tc)
names = ['GradientBoostingClassifier','MLPClassifier','DecisionTreeClassifier','RandomForestClassifier','LogisticRegression']

train_prep = D.Preprocessing(originaleTrain, prep_excluded=P.PreprocessingExcluded, scale=P.Scale,
	                     prep_included=P.PreprocessingIncluded)
train_prep.set_features_ordering(None)
test_prep = D.Preprocessing(originaleTest, prep_excluded=P.PreprocessingExcluded, scale=P.Scale,
	                    prep_included=P.PreprocessingIncluded)
test_prep.set_features_ordering(None)
test_prep.fit_transform()
train_ds = D.MotionSenseDataset(train_prep)
test_ds = D.MotionSenseDataset(test_prep)
train = train_ds.__inverse_transform_conv__(sensor_tensor=train_ds.sensor, phy=train_ds.phy_data, act_tensor=train_ds.activities, sens_tensor=train_ds.sensitive, user_id_tensor=train_ds.users_id, trials=train_ds.trials)
test = test_ds.__inverse_transform_conv__(sensor_tensor=test_ds.sensor, phy=test_ds.phy_data, act_tensor=test_ds.activities, sens_tensor=test_ds.sensitive, user_id_tensor=test_ds.users_id, trials=test_ds.trials)

xTrain = train.loc[:,'rotationRate.x':'userAcceleration.z']
xTrainOriginal = xTrain.as_matrix()
yTrainAct = train.loc[:,'act']
yTrainGender = train.loc[:,'gender']
yTrainAct = yTrainAct.as_matrix()
yTrainGender = yTrainGender.as_matrix()
xTest = test.loc[:,'rotationRate.x':'userAcceleration.z']
xTestOriginal = xTest.as_matrix()
yTestAct = test.loc[:,'act']
yTestGender = test.loc[:,'gender']
yTestAct = yTestAct.as_matrix()
yTestGender = yTestGender.as_matrix()
yTrainOriginal = yTrainAct
yTestOriginal = yTestAct

epoch = ['epoch']
accuAct =[['Act_GradientBoostingClassifierAccuracy'],['Act_MLPClassifierAccuracy'],['Act_DecisionTreeClassifierAccuracy'],['Act_RandomForestClassifierAccuracy'],['Act_LogisticRegressionAccuracy']]
accuGender =[['Gen_GradientBoostingClassifierAccuracy'],['Gen_MLPClassifierAccuracy'],['Gen_DecisionTreeClassifierAccuracy'],['Gen_RandomForestClassifierAccuracy'],['Gen_LogisticRegressionAccuracy']]
PredicActAccu = ['Act_PredicAccuracy']
DiscrimGenAccu = ['Gen_DiscrimAccuracy']
distance = ['distance']




for e in epochs:
	print("********** Epoch: " + str(e) + " **********")

	adressTrain = adressSan + "train_A=" + P.Alpha +"-L="+ P.Lambda +"-O=vector-KP="+ P.KPred +"-KD="+ P.KDisc +"-NN=2-Rec=Off-E="+ str(e) +".csv"
	adressTest = adressSan + "test_A=" + P.Alpha +"-L="+ P.Lambda +"-O=vector-KP="+ P.KPred +"-KD="+ P.KDisc +"-NN=2-Rec=Off-E="+ str(e) +".csv"
	features.features_creation(adressSave, adressTrain, adressTest,e)

	featuresTrain = adressSave + "/features_Train_"+ str(e) +".csv"
	featuressTest = adressSave + "/features_Test_"+ str(e) +".csv"

	train = pd.read_csv(featuresTrain)
	test = pd.read_csv(featuressTest)

	xTrain = train.loc[:,'X1train_mean':'Z2trainF_kurt']
	xTrain = xTrain.as_matrix()

	yTrainAct = train.loc[:,'act']
	yTrainGender = train.loc[:,'gender']
	yTrainAct = yTrainAct.as_matrix()
	yTrainGender = yTrainGender.as_matrix()

	xTest = test.loc[:,'X1test_mean':'Z2testF_kurt']
	xTest = xTest.as_matrix()

	yTestAct = test.loc[:,'act']
	yTestGender = test.loc[:,'gender']
	yTestAct = yTestAct.as_matrix()
	yTestGender = yTestGender.as_matrix()

	yTrain1 = yTrainAct
	yTest1 = yTestAct
	yTrain2 = yTrainGender
	yTest2 = yTestGender

	testBrute = pd.read_csv(adresseTest)
	xTestBrute = testBrute.loc[:,'rotationRate.x':'userAcceleration.z']
	xTestBrute = xTestBrute.as_matrix()
	tempDistance = []
	for j in range(len(xTestBrute[0])):
		tempDistance.append(Di.euclidean(xTestOriginal[:,j], xTestBrute[:,j]))
	distance.append(np.mean(tempDistance))

	epoch.append(e)
	
	for i in range(numberClass) :

		clf = tc[i]
		clf.fit(xTrain, yTrain1)
		predictions = clf.predict(xTest)
		accuAct[i].append(Me.Metrics().accuracy(predictions, yTest1))

		clf = tc[i]
		clf.fit(xTrain, yTrain2)
		predictions = clf.predict(xTest)
		accuGender[i].append(Me.Metrics().accuracy(predictions, yTest2))

	train_prep = D.Preprocessing(adresseTrain, prep_excluded=P.PreprocessingExcluded, scale=P.Scale,
		                     prep_included=P.PreprocessingIncluded, numeric_as_categoric_max_thr = 0)
	train_prep.set_features_ordering(None)
	test_prep = D.Preprocessing(adresseTest, prep_excluded=P.PreprocessingExcluded, scale=P.Scale,
		                    prep_included=P.PreprocessingIncluded, numeric_as_categoric_max_thr = 0)
	test_prep.set_features_ordering(None)
	test_prep.fit_transform()

	train_ds = D.MotionSenseDataset(train_prep)
	test_ds = D.MotionSenseDataset(test_prep)

	train_dl = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
	test_dl = data.DataLoader(test_ds, batch_size=batch_size, shuffle=True,num_workers=4)

	# Defining model Predicting activities.
	activities = np.unique(train_ds.activities)
	phys_shape = train_ds.phy_data.shape[1]

	model = M.PredictorConv(input_channels=train_ds.input_channels, seq_len=train_ds.seq_len, output_size=activities.shape[0], physNodes=phys_shape)
	# Send model on GPU or CPU
	model.to(device)
	# Loss
	# loss = torch.nn.NLLLoss()
	# loss = Cl.NLLLoss()
	loss = Cl.BalancedErrorRateLoss(targetBer=0, device=device)
	# loss = Cl.AccuracyLoss(device=device)
	# Training procedure
	max_epochs = 200
	losses = []
	t_key = "act"
	#t_key = "sens"

	for i in tqdm.tqdm(range(max_epochs)):
		# print("Epoch: {}".format(i))
		# set model to train and initialize aggregation variables
		model.train()
		total, sum_loss = 0, 0
		# for each batch
		# get the optimizer (allows for changing learning rates)
		optim = M.get_optimizer(model, wd=0.0005)
		for sample in train_dl:
			# get the optimizer (allows for changing learning rates)
			# optim = Models.get_optimizer(model, wd=0.00001)
			# put each of the batch objects on the device
			x = sample['sensor'].to(device)
			p = sample["phy"].to(device)
			s = sample["sens"].to(device)
			# u = sample["uid"].to(device)
			# y = sample['act'].unsqueeze(1).to(device)
			y = sample[t_key].to(device)
			yp = model(x, p)
			l = loss(yp, y, s)
			# l = loss(yp, s, s)
			optim.zero_grad()
			l.backward()
			optim.step()
			# print(l.data)
			sum_loss += l
			losses.append(l.to("cpu").data.numpy().tolist())
		# print(sum_loss.data)
	#plt.figure(figsize=(14, 14))
	#sns.lineplot(x=range(len(losses)), y=losses)
	#plt.savefig("TestLossesActivities.png")
	# Test phase:
	accAct = 0
	model.train(False)
	for sample in test_dl:
		x = sample['sensor'].to(device)
		p = sample["phy"].to(device)
		# y = sample['act'].to(device)
		y = sample[t_key].to(device)
		yp = model(x, p).argmax(1)
		try:
			accAct += np.abs((y == yp).data.numpy()).sum()
		except TypeError:
			accAct += np.abs((y.cpu() == yp.cpu()).data.numpy()).sum()
	accAct = accAct / test_ds.length
	print("Accuracy: {}".format(accAct))

	PredicActAccu.append(accAct)


	# Defining model Predicting activities.
	activities = np.unique(train_ds.sensitive)
	phys_shape = train_ds.phy_data.shape[1]


	#Parameters

	model = M.DiscriminatorConv(input_channels=train_ds.input_channels, seq_len=train_ds.seq_len, output_size=activities.shape[0], physNodes=phys_shape)
	# Send model on GPU or CPU
	model.to(device)
	# Loss
	# loss = torch.nn.NLLLoss()
	# loss = Cl.NLLLoss()
	loss = Cl.BalancedErrorRateLoss(targetBer=0, device=device)
	# loss = Cl.AccuracyLoss(device=device)
	# Training procedure
	max_epochs = 200
	losses = []
	#t_key = "act"
	t_key = "sens"
	print("Starting Training")
	for i in tqdm.tqdm(range(max_epochs)):
		# print("Epoch: {}".format(i))
		# set model to train and initialize aggregation variables
		model.train()
		total, sum_loss = 0, 0
		# for each batch
		# get the optimizer (allows for changing learning rates)
		optim = M.get_optimizer(model, wd=0.0005)
		for sample in train_dl:
			# get the optimizer (allows for changing learning rates)
			# optim = Models.get_optimizer(model, wd=0.00001)
			# put each of the batch objects on the device
			x = sample['sensor'].to(device)
			p = sample["phy"].to(device)
			s = sample["sens"].to(device)
			# u = sample["uid"].to(device)
			# y = sample['act'].unsqueeze(1).to(device)
			y = sample[t_key].to(device)
			yp = model(x, p)
			l = loss(yp, y, s)
			# l = loss(yp, s, s)
			optim.zero_grad()
			l.backward()
			optim.step()
			# print(l.data)
			sum_loss += l
			losses.append(l.to("cpu").data.numpy().tolist())
		# print(sum_loss.data)
	#plt.figure(figsize=(14, 14))
	#sns.lineplot(x=range(len(losses)), y=losses)
	#plt.savefig("TestLossesGender.png")
	# Test phase:
	accGen = 0
	model.train(False)
	for sample in test_dl:
		x = sample['sensor'].to(device)
		p = sample["phy"].to(device)
		# y = sample['act'].to(device)
		y = sample[t_key].to(device)
		yp = model(x, p).argmax(1)
		try:
			accGen += np.abs((y == yp).data.numpy()).sum()
		except TypeError:
			accGen += np.abs((y.cpu() == yp.cpu()).data.numpy()).sum()
	accGen = accGen / test_ds.length
	print("Accuracy: {}".format(accGen))

	DiscrimGenAccu.append(accGen)
        
with open(P.ExperimentBaseDir + P.SetName + "/Exp_" + P.ExpNumber  + "/results_classifiers.csv", "w") as f:
	writer = csv.writer(f)

	for a in range(len(epoch)):
		row = [epoch[a], distance[a]]
		row.append(PredicActAccu[a])
		row.append(DiscrimGenAccu[a])
		for i in range(numberClass):
			row.append(accuAct[i][a])
			row.append(accuGender[i][a])
		writer.writerow(row)
