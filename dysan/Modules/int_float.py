
import tqdm
import torch
from Parameters import Parameters as P
from Modules import Setup as S
from Modules import Models as M
from Modules import Results as R
from Modules import Metrics as Me
from Modules import Datasets as D
from Modules import CustomLosses as Cl
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

# Add timing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.cuda.empty_cache()

expe = "lr"
alpha = "0.5"
lambd = "0.4"
epochs = [1]
KP = "50"
KD = "50"

for e in epochs:
#for e in range(1,numberEpoch+1):
	print("********** Epoch: " + str(e) + " **********")

	adresseTrain = "/home/jourdan/Documents/motiongan/Experiments/Motion-Sense/Exp_test_" + expe + "/SubExp_1/Generated/train_A=" + alpha +"-L="+ lambd +"-O=vector-KP="+ KP +"-KD="+ KD +"-NN=2-Rec=On-E="+ str(e) +".csv"
	adresseTest = "/home/jourdan/Documents/motiongan/Experiments/Motion-Sense/Exp_test_" + expe + "/SubExp_1/Generated/test_A=" + alpha +"-L="+ lambd +"-O=vector-KP="+ KP +"-KD="+ KD +"-NN=2-Rec=On-E="+ str(e) +".csv"

	featuresTrain = "/home/jourdan/Documents/motiongan/Experiments/Motion-Sense/Exp_test_" + expe + "/features/features_Train_"+ str(e) +".csv"
	featuressTest = "/home/jourdan/Documents/motiongan/Experiments/Motion-Sense/Exp_test_" + expe + "/features/features_Test_"+ str(e) +".csv"


	#Pour les données features
	train = pd.read_csv(adresseTrain)
	test = pd.read_csv(adresseTest)

	xTrain = str(train.loc[:,'weight':'age'])
	xTrain = str(xTrain.as_matrix())
	print(type(xTrain))
	print(xTrain)
