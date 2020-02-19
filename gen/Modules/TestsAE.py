import torch
import numpy as np
from torch.utils import data
from Modules import Datasets
from Modules import Models

# Add timing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.cuda.empty_cache()

# Take the first 70% timestep as training.
train_ds = Datasets.MotionSenseDataset("../../Data/Csv/motion-sense-train.csv")
test_ds = Datasets.MotionSenseDataset("../../Data/Csv/motion-sense-test.csv")


# build data loaders
batch_size = 256
# Tester randomSampler to see if it only shuffle indices and not content
train_dl = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
test_dl = data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,num_workers=4)

# Defining model Predicting activities.
activities = np.unique(train_ds.activities)
phys_shape = train_ds.phy_data.shape[1]
model = Models.SanitizerConv(input_channels=train_ds.input_channels, seq_len=train_ds.seq_len, phyNodes=0)
# Send model on GPU or CPU
model.to(device)

# Loss
# loss = torch.nn.MSELoss()
# sum(0) will reduce from 3 to 2 dimensions
loss = lambda x, y: torch.nn.MSELoss(reduction="none")(x, y).sum(0).sum(1) / (x.size(0) * x.size(2))
# loss = torch.nn.L1Loss()
# Training procedure
max_epochs = 5000
for i in range(max_epochs):

    print("Epoch: {}".format(i))
    # set model to train and initialize aggregation variables
    model.train()
    total, sum_loss = 0, 0

    # for each batch
    # get the optimizer (allows for changing learning rates)
    optim = Models.get_optimizer(model, wd=0.00001)
    for sample in train_dl:
        # get the optimizer (allows for changing learning rates)
        # optim = Models.get_optimizer(model, wd=0.00001)


        # put each of the batch objects on the device
        x = sample['sensor'].to(device)
        p = sample["phy"].to(device)
        # s = sample["sens"].to(device)
        # u = sample["uid"].to(device)
        # y = sample['act'].unsqueeze(1).to(device)
        # y = sample['act'].to(device)
        xp, pp = model(x, p)
        l = loss(xp, x)
        if pp is not None:
            l += loss(pp, p)
        for e in l:
            optim.zero_grad()
            e.backward(retain_graph=True)
            optim.step()
        print(l.mean().data)

        sum_loss += l

    print(sum_loss.data)


# Test phase:
tsloss = 0
model.train(False)
for sample in test_dl:
    x = sample['sensor'].to(device)
    p = sample["phy"].to(device)
    # y = sample['act'].to(device)

    xp, pp = model(x, p)
    tsloss += loss(xp, x)
    # try:
    #     acc += np.abs((y == yp).data.numpy()).sum()
    # except TypeError:
    #     acc += np.abs((y.cpu() == yp.cpu()).data.numpy()).sum()



print("Final Loss: {}".format(tsloss.cpu().data.numpy()))


# 10000: 67
# 1000: 85
# 256: 88
# 64: 89
# 32: 86
# 2:

# Sum MSE: Final Loss: 8.896578788757324