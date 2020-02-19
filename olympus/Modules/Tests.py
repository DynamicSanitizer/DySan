import tqdm
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils import data
from Modules import ModelsFinal as M
from Modules import Datasets as D
from Modules import Parameters as P
from Modules import CustomLosses as Cl

# Add timing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.cuda.empty_cache()

# Take the first 70% timestep as training.
train_prep = D.Preprocessing("./" + P.TrainPath, prep_excluded=P.PreprocessingExcluded, scale=P.Scale,
                             prep_included=P.PreprocessingIncluded)
train_prep.set_features_ordering(None)
#train_prep.fit_transform() #ici, si on met en commentaire, on a la prediction sur les donnees brutes
test_prep = D.Preprocessing("./"+ P.TestPath, prep_excluded=P.PreprocessingExcluded, scale=P.Scale,
                            prep_included=P.PreprocessingIncluded)
test_prep.set_features_ordering(None)
test_prep.fit_transform()
train_ds = D.MotionSenseDataset(train_prep)
test_ds = D.MotionSenseDataset(test_prep)
# train_ds = Datasets.MotionSenseDataset("../../Data/Csv/motion-sense-train.csv")
# test_ds = Datasets.MotionSenseDataset("../../Data/Csv/motion-sense-test.csv")


# build data loaders
batch_size = 256
# Tester randomSampler to see if it only shuffle indices and not content
train_dl = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
test_dl = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,num_workers=4)



# Defining model Predicting activities.
activities = np.unique(train_ds.activities)
phys_shape = train_ds.phy_data.shape[1]


###################### ACTIVITIES ###############################
#Parameters

model = M.Predictor(input_channels=train_ds.input_channels, seq_len=train_ds.seq_len, output_size=activities.shape[0], physNodes=phys_shape)
# Send model on GPU or CPU
model.to(device)
# Loss
# loss = torch.nn.NLLLoss()
# loss = Cl.NLLLoss()
loss = Cl.BalancedErrorRateLoss(targetBer=0, device=device)
# loss = Cl.AccuracyLoss(device=device)
# Training procedure
max_epochs = 300
losses = []
t_key = "act"
#t_key = "sens"
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
plt.figure(figsize=(14, 14))
sns.lineplot(x=range(len(losses)), y=losses)
plt.savefig("TestLossesActivities.png")
# Test phase:
acc = 0
model.train(False)
for sample in test_dl:
    x = sample['sensor'].to(device)
    p = sample["phy"].to(device)
    # y = sample['act'].to(device)
    y = sample[t_key].to(device)
    yp = model(x, p).argmax(1)
    try:
        acc += np.abs((y == yp).data.numpy()).sum()
    except TypeError:
        acc += np.abs((y.cpu() == yp.cpu()).data.numpy()).sum()
acc = acc / test_ds.length
print("Accuracy: {}".format(acc))



##################### GENDER ############################



# Defining model Predicting activities.
activities = np.unique(train_ds.sensitive)
phys_shape = train_ds.phy_data.shape[1]


#Parameters

model = M.Discriminator(input_channels=train_ds.input_channels, seq_len=train_ds.seq_len, output_size=activities.shape[0], physNodes=phys_shape)
# Send model on GPU or CPU
model.to(device)
# Loss
# loss = torch.nn.NLLLoss()
# loss = Cl.NLLLoss()
loss = Cl.BalancedErrorRateLoss(targetBer=0, device=device)
# loss = Cl.AccuracyLoss(device=device)
# Training procedure
max_epochs = 300
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
plt.figure(figsize=(14, 14))
sns.lineplot(x=range(len(losses)), y=losses)
plt.savefig("TestLossesGender.png")
# Test phase:
acc = 0
model.train(False)
for sample in test_dl:
    x = sample['sensor'].to(device)
    p = sample["phy"].to(device)
    # y = sample['act'].to(device)
    y = sample[t_key].to(device)
    yp = model(x, p).argmax(1)
    try:
        acc += np.abs((y == yp).data.numpy()).sum()
    except TypeError:
        acc += np.abs((y.cpu() == yp.cpu()).data.numpy()).sum()
acc = acc / test_ds.length
print("Accuracy: {}".format(acc))