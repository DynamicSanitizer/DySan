import tqdm
import torch
from Parameters import Parameters as P
from Modules import Setup as S
from Modules import Models as M
from Modules import Results as R
from Modules import Metrics as Me
from Modules import Datasets as D
from Modules import CustomLosses as Cl
from Modules import PreProcessing
import pandas as pd
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
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")
torch.cuda.empty_cache()


def shaping(path):
# Set the data as the same for all, the generated ones and the original ones such that we have the same
# computation graph.
	data = D.MotionSenseDataset(path, window_overlap=P.Window_overlap)
	return data.__inverse_transform_conv__(sensor_tensor=data.sensor, phy=data.phy_data, act_tensor=data.activities, sens_tensor=data.sensitive,user_id_tensor=data.users_id, trials=data.trials, cpu_device=CPU_DEVICE)

params = ["0.1_0.1","0.1_0.2","0.1_0.3","0.1_0.4","0.1_0.5","0.1_0.6","0.1_0.7","0.1_0.8","0.2_0.1","0.2_0.2","0.2_0.3","0.2_0.4","0.2_0.5",
"0.2_0.6","0.2_0.7","0.3_0.1","0.3_0.2","0.3_0.3","0.3_0.4","0.3_0.5","0.3_0.6","0.4_0.1","0.4_0.2","0.4_0.3","0.4_0.4","0.4_0.5","0.5_0.1",
"0.5_0.2","0.5_0.3","0.5_0.4","0.6_0.1","0.6_0.2","0.6_0.3","0.7_0.1","0.7_0.2","0.8_0.1"]


#expe = sys.argv[1] + sys.argv[2]
#alpha = sys.argv[1]
#lambd = sys.argv[2]
#thebatch = int(sys.argv[3])

thebatch = 5


KP = "50"
KD = "50"
epoch = 1
max_epochs = 300


seed = 42
batch_size = 256


for parameter in params:
	
	alpha = parameter[0:3]
	lambd = parameter[4:7]
	expe = alpha + lambd
	
	print(alpha)
	print(lambd)

	adresseTrain = P.ExperimentBaseDir + P.SetName + "/Exp_" + expe + "/SubExp_1/Generated/train_A=" + alpha +"-L="+ lambd +"-O=vector-KP="+ KP +"-KD="+ KD +"-NN=2-Rec=On-E="+ str(epoch) +".csv"
	adresseTest =  P.ExperimentBaseDir + P.SetName + "/Exp_" + expe + "/SubExp_1/Generated/test_A=" + alpha +"-L="+ lambd +"-O=vector-KP="+ KP +"-KD="+ KD +"-NN=2-Rec=On-E="+ str(epoch) +".csv"

	saveAdresse1 = "temporary_"+str(thebatch)+"/"
	if not os.path.exists(saveAdresse1):
		oldmask = os.umask(000)
		os.mkdir(saveAdresse1, mode = 0o777)
		os.umask(oldmask)
	saveAdresse = saveAdresse1 + "exp_"+alpha+"_"+lambd
	if not os.path.exists(saveAdresse):
		oldmask = os.umask(000)
		os.mkdir(saveAdresse, mode = 0o777)
		os.umask(oldmask)

	accur = []
	gend = []


	print("** Activities **")
	train_prep = D.Preprocessing(adresseTrain, prep_excluded=P.PreprocessingExcluded, scale=P.Scale,
				     prep_included=P.PreprocessingIncluded, numeric_as_categoric_max_thr = 0)
	train_prep.set_features_ordering(None)
	train_ds = D.MotionSenseDataset(train_prep)
	train_dl = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

	activities = np.unique(train_ds.activities)
	phys_shape = train_ds.phy_data.shape[1]
	model = M.PredictorConv(input_channels=train_ds.input_channels, seq_len=train_ds.seq_len, output_size=activities.shape[0], physNodes=phys_shape)

	model.to(device)
	loss = Cl.BalancedErrorRateLoss(targetBer=0, device=device)
	losses = []
	t_key = "act"
	#t_key = "sens"

	if os.path.isfile(saveAdresse+"/Predictor_"+alpha+"_"+lambd+"_Model") :
		M.load_classifier_state2(model,saveAdresse+"/Predictor_"+alpha+"_"+lambd)
	else:

		print("The model has never been created and trained before.")
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
		M.save_classifier_states2(model, "Predictor_"+alpha+"_"+lambd)

	accAct = 0

	model.train(False)

	for e in range(54):
		#for e in range(1,numberEpoch+1):
		print("********** Identity: " + str(e) + " **********")

		testBrute = pd.read_csv(adresseTest)
		b = testBrute.loc[testBrute["id"] == e]
		b = b.reset_index().drop(columns='index')
		b.to_csv('temporary_file.csv', index = False)
		temp = 'temporary_file.csv'


		a = torch.load('./Modules/rf_mobiact.pt')
		truepred = a.activities.numpy()
		truepredid = a.users_id.numpy()
		locid = np.where(truepredid == e)[0]
		activitychange = truepred[locid]



		test_prep = D.Preprocessing(temp, prep_excluded=P.PreprocessingExcluded, scale=P.Scale,
					    prep_included=P.PreprocessingIncluded, numeric_as_categoric_max_thr = 0)
		test_prep.set_features_ordering(None)
		test_prep.fit_transform()
		test_ds = D.MotionSenseDataset(test_prep)
		for actc in range(len(activitychange)):
			test_ds.activities.numpy()[actc] = activitychange[actc]
		test_dl = data.DataLoader(test_ds, batch_size=test_ds.length, shuffle=False,num_workers=4)
		curseur = 0
		for sample in test_dl:
			longue = len(sample['act'])
			cur = 0
			while 5+cur <= longue+1:

				x = sample['sensor'][0+cur:5+cur].to(device)
				p = sample['phy'][0+cur:5+cur].to(device)
				sensit = sample['sens'][0+cur:5+cur].to(device)
				# y = sample['act'].to(device)
				y = sample[t_key][0+cur:5+cur].to(device)
				yp = model(x, p).argmax(1)
				#print(y)
				try:
					accAct = np.abs((y == yp).data.numpy()).sum()
				except TypeError:
					accAct = np.abs((y.cpu() == yp.cpu()).data.numpy()).sum()

				curseur += 1
				cur += 5
				accur_cur.append(accAct / len(x))
		with open(saveAdresse +'/activity_id_'+str(e)+'.csv', 'w') as f:
			writer = csv.writer(f)
			for a in range(len(accur_cur)):
				writer.writerow([accur_cur[a]])


	print("** Gender **")

	# Defining model Predicting activities.
	activities = np.unique(train_ds.sensitive)
	phys_shape = train_ds.phy_data.shape[1]


	#Parameters

	model = M.DiscriminatorConv(input_channels=train_ds.input_channels, seq_len=train_ds.seq_len, output_size=activities.shape[0], physNodes=phys_shape)
	#model = M.PredictorConv(input_channels=train_ds.input_channels, seq_len=train_ds.seq_len, output_size=activities.shape[0], physNodes=phys_shape)

	# Send model on GPU or CPU
	model.to(device)
	# Loss
	# loss = torch.nn.NLLLoss()
	# loss = Cl.NLLLoss()
	loss = Cl.BalancedErrorRateLoss(targetBer=0, device=device)
	# loss = Cl.AccuracyLoss(device=device)
	# Training procedure

	losses = []
	#t_key = "act"
	t_key = "sens"

	if os.path.isfile(saveAdresse+"/Discriminator_"+alpha+"_"+lambd+"_Model") :
		M.load_classifier_state2(model,saveAdresse+"/Discriminator_"+alpha+"_"+lambd)
	else:
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
				optim.zero_grad()
				l.backward()
				optim.step()
				sum_loss += l
				losses.append(l.to("cpu").data.numpy().tolist())
		M.save_classifier_states2(model, "Discriminator_"+alpha+"_"+lambd)
	accGen = 0
	model.train(False)

	for e in range(54):
	#for e in range(1,numberEpoch+1):
		print("********** Identity: " + str(e) + " **********")

		#Pour les données brutes
		pred_cur = ['Gender']

		testBrute = pd.read_csv(adresseTest)
		b = testBrute.loc[testBrute["id"] == e]
		b = b.reset_index().drop(columns='index')
		b.to_csv('temporary_file_.csv', index = False)
		temp = 'temporary_file_.csv'


		test_prep = D.Preprocessing(temp, prep_excluded=P.PreprocessingExcluded, scale=P.Scale,
		prep_included=P.PreprocessingIncluded, numeric_as_categoric_max_thr = 0)

		test_prep.set_features_ordering(None)
		test_prep.fit_transform()
		test_ds = D.MotionSenseDataset(test_prep)
		#test_dl = data.DataLoader(test_ds, batch_size=int(test_ds.length/4 + 0.75), shuffle=False,num_workers=4)
		test_dl = data.DataLoader(test_ds, batch_size=test_ds.length, shuffle=False,num_workers=4)
		curseur = 0
		for sample in test_dl:

			longue = len(sample['act'])
			cur = 0
			curseur += 1
			while 5+cur <= longue+1:
				x = sample['sensor'][0+cur:5+cur].to(device)
				p = sample['phy'][0+cur:5+cur].to(device)
				y = sample[t_key][0+cur:5+cur].to(device)
				yp = model(x, p).argmax(1)
				try:
					accGen = np.abs((y == yp).data.numpy()).sum()

				except TypeError:
					accGen = np.abs((y.cpu() == yp.cpu()).data.numpy()).sum()
				cur += 5
				pred_cur.append(accGen / len(x))
		with open(saveAdresse +'/gender_id_'+str(e)+'.csv', 'w') as f:
			writer = csv.writer(f)
			for a in range(len(pred_cur)):
				row = []
				row.append(pred_cur[a])
				writer.writerow(row)

######################################################################
# Selection of the best models


numberDec = 5
poid1 = 0.2
poid2 = 0.8
#numberDec = int(sys.argv[1])
#poid1 = float(sys.argv[2])
#poid2 = float(sys.argv[3])

from sklearn.metrics import accuracy_score
from scipy.spatial import distance as Di
exp = ""
adre = 'temporary_'+str(numberDec)+exp+'/resultats/'
if not os.path.exists(adre):
	oldmask = os.umask(000)
	os.mkdir(adre, mode = 0o777)
	os.umask(oldmask)


#ids = [*range(0, 24, 1)] 
ids = [*range(0, 54, 1)] 
valuesAct = []
valuesGen = []
hyper = []
ident = []
actperid = []
genperid = []

for user in ids:
	longu = len(pd.read_csv("decoupe_"+str(numberDec)+exp+"/exp_0.1_0.1/activity_id_"+str(user)+".csv").loc[:,"Activity"].values)
	df = pd.DataFrame(index=range(len(params)),columns=range(longu))
	dfAct = pd.DataFrame(index=range(len(params)),columns=range(longu))
	dfGen = pd.DataFrame(index=range(len(params)),columns=range(longu))
	dfRaw = pd.DataFrame(index=range(len(params)),columns=range(longu))
	dfActval = pd.DataFrame(index=range(len(params)),columns=range(longu))
	dfGenval = pd.DataFrame(index=range(len(params)),columns=range(longu))

	localAct = []
	localGen = []

	for par in range(len(params)):
		data_act1 = pd.read_csv("decoupe_"+str(numberDec)+exp+"/exp_"+params[par]+"/activity_id_"+str(user)+".csv")
		data_gen1 = pd.read_csv("decoupe_"+str(numberDec)+exp+"/exp_"+params[par]+"/gender_id_"+str(user)+".csv")
		data_act = data_act1.loc[:,"Activity"].values
		data_gen = data_gen1.loc[:,"Gender"].values
		#data_raw = data_act1.loc[:,"Raw"].values
		#data_actval = data_act1.loc[:,"Act"].values
		#data_genval = data_act1.loc[:,"Gen"].values
		

		for fen in range(longu):
			act = data_act[fen]
			gen = data_gen[fen]

			priv = 1 - abs(0.5-gen)

			df.loc[par,fen] = poid1*act + poid2*priv

			dfAct.loc[par,fen] = act
			dfGen.loc[par,fen] = gen
			

	for fen in range(longu):
		#print(df.loc[:,fen])
		#indi = df.loc[:,fen].idxmax()
		indi = np.argmax(df.loc[:,fen].values)
		#print(indi)
		hyper.append(params[indi])
		valuesAct.append(dfAct.loc[indi,fen])
		valuesGen.append(dfGen.loc[indi,fen])

		#a = dfRaw.loc[indi,fen][:]

		ident.append(user)

		localAct.append(dfAct.loc[indi,fen])
		localGen.append(dfGen.loc[indi,fen])

	df.to_csv(adre+'combi_'+str(user)+'.csv', index = False)
	dfAct.to_csv(adre+'combi_act_'+str(user)+'.csv', index = False)
	dfGen.to_csv(adre+'combi_gen_'+str(user)+'.csv', index = False)

	actperid.append(np.mean(localAct))
	genperid.append(np.mean(localGen))

data = np.array([ident,hyper,valuesAct,valuesGen])
datap = pd.DataFrame(data)
datap.to_csv(adre+'results_tab.csv')
print(np.mean(valuesAct))
print(np.mean(valuesGen))


