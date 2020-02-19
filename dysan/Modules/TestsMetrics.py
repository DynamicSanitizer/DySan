import tqdm
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils import data
from Parameters import Parameters as P
from Modules import Datasets as D
from Modules import Metrics as Me
from Modules import CustomLosses as Cl

# Add timing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.cuda.empty_cache()

# Take the first 70% timestep as training.
train_prep = D.Preprocessing("../" + P.TrainPath, prep_excluded=P.PreprocessingExcluded, scale=P.Scale,
                             prep_included=P.PreprocessingIncluded)
train_prep.set_features_ordering(None)
train_prep.fit_transform()
test_prep = D.Preprocessing("../"+ P.TestPath, prep_excluded=P.PreprocessingExcluded, scale=P.Scale,
                            prep_included=P.PreprocessingIncluded)
test_prep.set_features_ordering(None)
test_prep.fit_transform()
train_ds = D.MotionSenseDataset(train_prep)
test_ds = D.MotionSenseDataset(test_prep)


# build data loaders
batch_size = 256
# Defining model Predicting activities.
activities = np.unique(train_ds.activities)
phys_shape = train_ds.phy_data.shape[1]


metrics = Me.Metrics(input_channels=train_ds.input_channels, seq_len=train_ds.seq_len, kernel_sizes=[5,5],
                     strides=[1, 1], conv_paddings=[0, 0], gpu=True,
                     gpu_device="cuda:0", prep_included=P.PreprocessingIncluded, prep_excluded=P.PreprocessingExcluded,
                     scale=P.Scale, features_ordering=None, seed=42, verbose=True)

# sp = metrics.sensitive_attribute(train_set=pd.read_csv("../" + P.TrainPath), test_set=pd.read_csv("../" + P.TestPath),
#                                  use_accuracy=False, drop=[], sens_name="sens", phys_names="phy", epoch=200, batch_size=batch_size,
#                                  loss_fn=None, verbose=False)
# tp = metrics.sensitive_attribute(train_set="../" + P.TrainPath, test_set="../" + P.TestPath, use_accuracy=False,
#                                  drop=[], sens_name="sens", phys_names="phy", epoch=200, batch_size=batch_size, loss_fn=Cl.AccuracyLoss(),
#                                  verbose=False)
tp = metrics.sensitive_attribute(train_set=pd.read_csv("../" + P.TrainPath), test_set=pd.read_csv("../" + P.TestPath),
                                 use_accuracy=False, act_name="act", drop=[], sens_name="gender", ms_act_name="act",
                                 ms_sens_name="sens", sklearn_data_process=None, use_phys=True, physNodes=3,
                                 phys_names=["height", "weight", "age"], ms_phys_name="phy", epoch=200, batch_size=256,
                                 loss_fn=None)

sp = metrics.task(train_set=pd.read_csv("../" + P.TrainPath), test_set=pd.read_csv("../" + P.TestPath),
                  act_name="act", drop=[], sens_name="gender", ms_act_name="act", ms_sens_name="sens",
                  sklearn_data_process=None, use_phys=True, physNodes=3, phys_names=["height", "weight", "age"],
                  ms_phys_name="phy", epoch=200, batch_size=256, loss_fn=None)

print(tp)
print(sp)
