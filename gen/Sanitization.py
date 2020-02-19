import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import time
import torch
import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
from torch.utils import data
from Parameters import Parameters as P
from Modules import Setup as S
from Modules import Models as M
from Modules import Results as R
from Modules import Metrics as Me
from Modules import Datasets as D
from Modules import CustomLosses as Cl



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CPU_DEVICE = torch.device("cpu")
torch.cuda.empty_cache()

print(P.ExpNumber)

import importlib
IM = lambda p, f: getattr(importlib.import_module(name=p), f)
_FN_ = IM(P.NoiseGenerator["p"], P.NoiseGenerator["f"])
NOISE = lambda r=P.BatchSize: _FN_(r, P.NoiseNodes)
# Cast tensor to floatTensor,  and reshape to view -1, 1. Mostly used for LongTensor (accuracy, user_id etc...)
CTF = lambda x: x.view(-1, 1).type(torch.FloatTensor).to(DEVICE)

P.ExpPath, P.PrmPath, counter = S.prepare_experiment(baseDir=P.ExperimentBaseDir, setBaseName=P.SetName,
                                                     expNumber=P.ExpNumber, paramsDirId=P.ParamsDirId)
def tryReading(setPath):
    """
    Read the given path, and make sure that the dataset has at least some rows in it.
    :param setPath: path to the dataset
    :return: True if can read, False if not
    """
    try:
        assert (pd.read_csv(setPath).size != 0), "DataFrame is empty"
        return True
    except (FileNotFoundError, pd.errors.EmptyDataError, AssertionError) as e:
        return False

def generate_dataset(san, train_prep, train_ds, test_prep, test_ds, gen_path, train_id="train",
                     test_id="test", max_epoch=0, addParamFn=None, phys=1, san_acts=1, san_phys=1):
    """
    Generate a new dataset for each epoch
    :param san: the sanitizer model
    :param train_prep: the preprocessing core for the train set
    :param train_dl: the train dataloader
    :param test_prep: the preprocessing core for the test set
    :param test_dl: the test data loader
    :param gen_path: the path where to save the generated dataset
    :param train_id: the id of the train set
    :param test_id: the id of the test set
    :param epoch: the current epoch
    :param addParamFn: function to add parameters in name
    """
    bs = 1024
    train_dl = data.DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=1)
    test_dl = data.DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=1)
    name = lambda n, e: "{}/{}_{}.csv".format(gen_path, n, addParamFn(e))
    def _generate_(san, dataloader, dset, prep, phys=phys, san_acts=san_acts, san_phys=san_phys):
        df = pd.DataFrame()
        select = lambda x, y, c: x if c else y
        cat = lambda x, ax: x if ax is None else torch.cat((ax, x), 0)
        a_xs, a_p, a_a, a_s, a_uid = None, None, None, None, None
        for i, sample in enumerate(dataloader):
            x = sample["sensor"].to(DEVICE)
            noise = NOISE(r=x.shape[0]).to(DEVICE)
            a = sample["act"].to(DEVICE)
            p = sample["phy"].to(DEVICE) * phys
            other_data = torch.cat((p, CTF(a)*san_acts, noise), 1)
            s = sample["sens"]
            xs, acs, ps = san(x, other_data)

            a = select(acs, a, san_acts == 1)
            p = select(ps, p, san_phys == 1)
            a_xs = cat(xs, a_xs)
            a_p = cat(p, a_p)
            a_a = cat(a, a_a)
            a_s = cat(s, a_s)
            a_uid = cat(sample["uid"], a_uid)

        df = dset.__inverse_transform_conv__(sensor_tensor=a_xs, phy=a_p, act_tensor=a_a, sens_tensor=a_s,
                                             user_id_tensor=a_uid, trials=dset.trials, cpu_device=CPU_DEVICE)
        p = prep.copy(True)
        p.df = df.reset_index(drop=True)
        p.inverse_transform()
        return p.df

    for epoch in tqdm.tqdm(range(1, max_epoch+1)):
        # Load the sanitizer Model
        # range(1, max_epoch+1)
        M.load_classifier_state(san, epoch, P.ModelsDir(), ext="S", otherParamFn=P.ParamFunction)
        # Check if there is a generated data in the correct format
        if not tryReading(name(train_id, epoch)):
            # Set does not exits
            df = _generate_(san, train_dl, train_ds, train_prep)
            df.to_csv(name(train_id, epoch), index=False)
        if not tryReading(name(test_id, epoch)):
            # Set does not exits
            df = _generate_(san, test_dl, test_ds, test_prep)
            df.to_csv(name(test_id, epoch), index=False)


def generate_if_missing(final_epoch, san, train_prep, train_dl, test_prep, test_dl, gen_path, train_id="train", test_id="test",
                         epoch=0, addParamFn=None, noise_fn=None):
    for epoch in range(1, final_epoch):
        pass
        # Read if false then write

def train_sanitizer(sample, san, disc, pred, loss, optim, act_fn, act_select, phys_select, phys=1, san_acts=1):
    """
    Train the sanitizer
    :param k: the number of iterations of the sanitizer
    :param phys: set to 1 if want to use the physio data. O If not. san_acts and disc_acts are similar.
    San act for sanitizing activities while disc_act for decorrelating activities and the sensitive attribute.
    :return: the computed loss and dataloader iterator
    """
    x = sample["sensor"].to(DEVICE)
    noise = NOISE(r=x.shape[0]).to(DEVICE)
    a = sample["act"].to(DEVICE)
    p = sample["phy"].to(DEVICE) * phys
    other_data = torch.cat((p, CTF(a)*san_acts, noise), 1)
    s = sample["sens"] # Do not send to device. Send it only if it is used in the loss

    xs, acs, ps = san(x, other_data)
    ap = pred(xs, ps)
    # Why should we include the sanitized physio and sanitized activities ?
    sp = disc(xs, act_fn(phys_select(p, ps), act_select(CTF(a), acs.argmax(1))))
    l = loss(sensor_s=xs, other_s=torch.cat((ps, acs*san_acts), 1), act_p=ap, sens_p=sp, sensor=x, act=a, sens=s,
             other=torch.cat((p, CTF(a)*san_acts), 1))
    for e in l[0].view(-1, 1):
        optim.zero_grad()
        e.backward(retain_graph=True)
        optim.step()

    return l

def train_predictor2(k, san, model, dl, dl_iter, loss, optim, act_fn, act_select, phys_select, target_key="act",
                    sens_key="sens", phys=1, san_acts=1):
    """
    Train the predictors, either the activity predictor or the discriminator
    :param k: number of iterations to train the model for
    :param model: the model to train.
    :param dl_iter: dataloader iterator
    :param act_fn: Function that will handle the concat step for the discriminator if deemed necessary
    :param act_select: Function to select either the sanitized or the original activity to gives as input of the disc.
    :param phys: set to 1 if want to use the physio data. O If not
    :return: the accumulated loss, the dl_iterator
    """
    totalLoss = 0
    for i in range(k):
        try:
            sample = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            sample = next(dl_iter)

        x = sample["sensor"].to(DEVICE)
        # Will be set to 0 if do not want to use the physio data. Will consume some memory, but faster than having
        # a lot of if-else
        noise = NOISE(r=x.shape[0]).to(DEVICE)
        p = sample["phy"].to(DEVICE) * phys
        a = sample["act"].to(DEVICE)
        o = torch.cat((p, CTF(a)*san_acts, noise), 1)
        s = sample[sens_key] # Do not send to device. Send it only if it is used in the loss
        y = sample[target_key].to(DEVICE)
        xs, acs, ps = san(x, o)
        yp = model(xs, act_fn(phys_select(p, ps), act_select(CTF(a), acs.argmax(1))))
        l = loss(yp, y, s)
        totalLoss += l
        optim.zero_grad()
        l.backward()
        optim.step()

    return totalLoss / k, dl_iter


def train_predictor(losses, k, san, model, dl, dl_iter, loss, optim, act_fn, act_select, phys_select, target_key="act",
                    sens_key="sens", phys=1, san_acts=1):
    """
    Train the predictors, either the activity predictor or the discriminator
    :param k: number of iterations to train the model for
    :param model: the model to train.
    :param dl_iter: dataloader iterator
    :param act_fn: Function that will handle the concat step for the discriminator if deemed necessary
    :param act_select: Function to select either the sanitized or the original activity to gives as input of the disc.
    :param phys: set to 1 if want to use the physio data. O If not
    :return: the accumulated loss, the dl_iterator
    """
    totalLoss = 0
    i = 0
    lossLocal = []
    if len(losses)>0:
        firstLoss = losses[0]
    else:
        firstLoss = 0.5

    for i in range(k):
        try:
            sample = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            sample = next(dl_iter)

        x = sample["sensor"].to(DEVICE)
        # Will be set to 0 if do not want to use the physio data. Will consume some memory, but faster than having
        # a lot of if-else
        noise = NOISE(r=x.shape[0]).to(DEVICE)
        p = sample["phy"].to(DEVICE) * phys
        a = sample["act"].to(DEVICE)
        o = torch.cat((p, CTF(a)*san_acts, noise), 1)
        s = sample[sens_key] # Do not send to device. Send it only if it is used in the loss
        y = sample[target_key].to(DEVICE)
        xs, acs, ps = san(x, o)
        yp = model(xs, act_fn(phys_select(p, ps), act_select(CTF(a), acs.argmax(1))))
        l = loss(yp, y, s)
        totalLoss += l
        optim.zero_grad()
        l.backward()
        optim.step()
        lossLocal.append(l.item())
        i+=1
        if i > 20 and  np.max(lossLocal[len(lossLocal)-20:len(lossLocal)])< (firstLoss/10):
            break

    #print(i)
    return totalLoss / k, dl_iter

# Set the parameters
def sanitization_generation_metrics(feature_order=None, alpha_=P.Alpha, lambda_=P.Lambda, san_loss=P.SanLoss, pred_loss=P.PredLoss,
                                    disc_loss=P.DiscLoss, max_epoch=P.Epoch, k_pred=P.KPred, k_disc=P.KDisc, scale=P.Scale):

    # Return models and datasets

    # Take the first 70% timestep as training.
    train_prep = D.Preprocessing(P.TrainPath, prep_excluded=P.PreprocessingExcluded, scale=P.Scale,
                                 prep_included=P.PreprocessingIncluded)
    train_prep.set_features_ordering(feature_order)
    train_prep.fit_transform()
    test_prep = D.Preprocessing(P.TestPath, prep_excluded=P.PreprocessingExcluded, scale=P.Scale,
                                prep_included=P.PreprocessingIncluded)
    test_prep.set_features_ordering(feature_order)
    test_prep.fit_transform()
    train_ds = D.MotionSenseDataset(train_prep, window_overlap=P.Window_overlap)
    test_ds = D.MotionSenseDataset(test_prep, window_overlap=P.Window_overlap)

    # Shape of unique values
    uniq_act = np.unique(train_ds.activities)
    uniq_sens = np.unique(train_ds.sensitive)
    uniq_uid = np.unique(train_ds.users_id)
    phys_cols = train_ds.phy_data.shape[1]
    try:
        act_cols = train_ds.activities.shape[1]
    except IndexError:
        act_cols = 1

    # Discriminator target
    disc_target_values = uniq_sens
    pred_target_values = uniq_act

    # Load dataset
    # Create dataloader
    # build data loaders
    batch_size = P.BatchSize
    s_train_dl = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    d_train_dl = data.DataLoader(train_ds.copy(True), batch_size=batch_size, shuffle=True, num_workers=4)
    d_dl_iter = iter(d_train_dl)
    p_train_dl = data.DataLoader(train_ds.copy(True), batch_size=batch_size, shuffle=True, num_workers=4)
    p_dl_iter = iter(p_train_dl)

    # Create models:

    sanitizer = M.SanitizerConv(input_channels=train_ds.input_channels, seq_len=train_ds.seq_len, kernel_sizes=[5, 5],
                                strides=[1, 1], conv_paddings=[0, 0], phyNodes=phys_cols, noiseNodes=P.NoiseNodes,
                                actNodes=act_cols)

    # Adding physio data can prevent the sensor information to be dependent of such attribute because the disc model
    # Can not predict the sensitive value even though the height weight and other are given. Or we know that if an
    # attribute is strongly correlated, then the model will find such correlation. Example: Create a model to predict
    # something and in train set, give the target as data input the predict the same target. The model will learn to dis-
    # regard other columns
    # Predictor model output should be of same shape as necessary for NLLLoss. (model output is a matrix while target is
    # a vector).
    def get_models(input_channels=train_ds.input_channels, seq_len=train_ds.seq_len,
                   pred_out_size=pred_target_values.shape[0], disc_out_size=disc_target_values.shape[0],
                   phys_cols=phys_cols, act_cols=act_cols):
        predictor = M.PredictorConv(input_channels=input_channels, seq_len=seq_len, output_size=pred_out_size,
                                    physNodes=phys_cols)
        M.load_classifier_state2(predictor, "Predictor")
        predictor.to(DEVICE)
        pred_optim = M.get_optimizer(predictor,)

        discriminator = M.DiscriminatorConv(input_channels=input_channels, seq_len=seq_len, output_size=disc_out_size,
                                            physNodes=phys_cols+act_cols)
        M.load_classifier_state2(discriminator,"Discriminator")
        discriminator.to(DEVICE)
        disc_optim = M.get_optimizer(discriminator,)
        return predictor, pred_optim, discriminator, disc_optim

    def reset_weights(m):
        try:
            m.reset_parameters()
        except AttributeError as e:
            pass
            # print(e)
            # print("Layer not affected")

    predictor, pred_optim, discriminator, disc_optim = get_models()
    # Send models on GPU or CPU
    sanitizer.to(DEVICE)

    # Check the latest Epoch to start sanitization
    start_epoch = M.get_latest_states(P.ModelsDir(), sanitizer, discriminator, predictor, otherParamFn=P.ParamFunction)

    # Initialise losses
    san_loss = Cl.SanitizerBerLoss(alpha_=alpha_, lambda_=lambda_, recOn=P.RecOn, optim_type=P.OptimType, device=DEVICE)
    # pred_loss = Cl.AccuracyLoss(device=DEVICE)
    pred_loss = Cl.BalancedErrorRateLoss(targetBer=0, device=DEVICE)
    disc_loss = Cl.BalancedErrorRateLoss(targetBer=0, device=DEVICE)

    # Optimizers
    san_optim = M.get_optimizer(sanitizer,)

    losses_frame_path = "{}/{}.csv".format(P.ModelsDir(), P.ParamFunction("losses"))
    san_losses = [
        [], [], []
    ]
    disc_losses = []
    pred_losses = []
    if (start_epoch > 1) and tryReading(losses_frame_path):
        losses_frame = pd.read_csv(losses_frame_path)
        disc_losses = losses_frame["disc"].values.tolist()
        pred_losses = losses_frame["pred"].values.tolist()
        san_losses = losses_frame.drop(["pred", "disc"], axis=1).T.values.tolist()

    # Function to differentiate and integrate the activities. (Ignore for the predictor, integrate for the sanitizer)
    act_fn_disc = lambda ps, act: torch.cat((ps, act*P.DecorrelateActAndSens), 1)
    act_fn_pred = lambda ps, act: ps

    # Init figure
    fig = "asdfoijbnad"
    plt.figure(fig, figsize=(14, 14))

    predictor.train(False)
    discriminator.train(False)
    
    # Sanitize
    print("Starting Sanitizing ......>")
    for epoch in tqdm.tqdm(range(start_epoch, max_epoch+1)):
        print("Current Epoch: {}".format(epoch))
        if P.TrainingResetModelsStates:
            predictor.apply(reset_weights)
            discriminator.apply(reset_weights)

            # del predictor
            # del discriminator
            # del disc_optim
            # del pred_optim
            # predictor, pred_optim, discriminator, disc_optim = get_models()

        for sample in s_train_dl:

            # Train the sanitizer
            l = train_sanitizer(sample, sanitizer, discriminator, predictor, san_loss,
                                           san_optim, act_fn=act_fn_disc, act_select=P.ActivitySelection,
                                           phys_select=P.PhysiologSelection, phys=P.PhysInput,
                                           san_acts=P.SanitizeActivities,)
            san_losses[0].append(l[0].mean().to(CPU_DEVICE).data.numpy().reshape(-1)[0])
            san_losses[1].append(l[1].to(CPU_DEVICE).data.numpy().reshape(-1)[0])
            san_losses[2].append(l[2].to(CPU_DEVICE).data.numpy().reshape(-1)[0])

            # Train the predictor
            #l, p_dl_iter = train_predictor(pred_losses,k_pred, sanitizer, predictor, p_train_dl, p_dl_iter, pred_loss, pred_optim,
            #                               act_fn=act_fn_pred, act_select=P.ActivitySelection,
            #                               phys_select=P.PhysiologSelection, target_key="act",
            #                               sens_key="sens", phys=P.PhysInput, san_acts=P.SanitizeActivities,)
            #pred_losses.append(l.to(CPU_DEVICE).data.numpy().reshape(-1)[0])
            pred_losses.append(1)
            # Train the discriminator
            #l, d_dl_iter = train_predictor(disc_losses,k_pred, sanitizer, discriminator, d_train_dl, d_dl_iter, disc_loss, disc_optim,
            #                               act_fn=act_fn_disc, act_select=P.ActivitySelection,
            #                               phys_select=P.PhysiologSelection, target_key="sens",
            #                               sens_key="sens", phys=P.PhysInput, san_acts=P.SanitizeActivities,)
            #disc_losses.append(l.to(CPU_DEVICE).data.numpy().reshape(-1)[0])
            disc_losses.append(1)

        print("***")
        # Save losses, and models states.
        # Saving models States.
        M.save_classifier_states(sanitizer, epoch, P.ModelsDir(), otherParamFn=P.ParamFunction, ext="S")
        M.save_classifier_states(discriminator, epoch, P.ModelsDir(), otherParamFn=P.ParamFunction, ext="D")
        M.save_classifier_states(predictor, epoch, P.ModelsDir(), otherParamFn=P.ParamFunction, ext="P")
        # Saving and plotting losses
        losses_frame = pd.DataFrame.from_dict({
            "san_rec": san_losses[0], "san_act": san_losses[1], "san_sens": san_losses[2],
            "disc": disc_losses, "pred": pred_losses,
        })
        losses_frame.to_csv(losses_frame_path, index=False)
        losses_frame["san_sens"] = san_loss.disc_loss.get_true_value(losses_frame["san_sens"].values)
        if epoch % P.PlotRate == 0:
            plt.subplot(5, 1, 1)
            sns.lineplot(x="index", y="san_rec", data=losses_frame.reset_index())
            plt.subplot(5, 1, 2)
            sns.lineplot(x="index", y="san_act", data=losses_frame.reset_index())
            plt.subplot(5, 1, 3)
            sns.lineplot(x="index", y="san_sens", data=losses_frame.reset_index())
            plt.subplot(5, 1, 4)
            sns.lineplot(x="index", y="disc", data=losses_frame.reset_index())
            plt.subplot(5, 1, 5)
            sns.lineplot(x="index", y="pred", data=losses_frame.reset_index())
            plt.savefig("{}/{}.png".format(P.FiguresDir(), P.ParamFunction("losses")))
            plt.clf()
        

    # Check datasets and generate
# def generate_dataset(san, train_prep, train_ds, train_dl, test_prep, test_ds, test_dl, gen_path, train_id="train",
#                      test_id="test", max_epoch=0, addParamFn=None, phys=1, san_acts=1, san_phys=1):
    
    print("Generating Sanitized Datasets")
    generate_dataset(sanitizer, train_prep, train_ds, test_prep, test_ds, P.GenDataDir(), max_epoch=P.Epoch,
                     addParamFn=P.ParamFunction, phys=P.PhysInput, san_acts=P.SanitizeActivities,
                     san_phys=P.SanitizePhysio)
    # Check if everything has been correctly generated

    #print("Computing Metrics")
    # If device == cpu_device, then we are not supposed to use gpu as there might not be anyone
    """metrics_computation(input_channels=train_ds.input_channels, seq_len=train_ds.seq_len,
                        features_order=feature_order, kernel_size=[5, 5], strides=[1, 1], conv_paddings=[0, 0],
                        gpu=DEVICE!=CPU_DEVICE, gpu_device=DEVICE, gen_path=P.GenDataDir(), alpha_=P.Alpha,
                        lambda_=P.Lambda, train_id="train", test_id="test", epoch=P.Epoch, addParamFn=P.ParamFunction,
                        seed=42, verbose=False)"""

def metrics_computation(input_channels, seq_len, features_order, kernel_size=[5, 5], strides=[1, 1], conv_paddings=[0, 0],
                        gpu=True, gpu_device="cuda:0", gen_path=P.GenDataDir(), alpha_=P.Alpha, lambda_=P.Lambda,
                        train_id="train", test_id="test", epoch=P.Epoch, addParamFn=P.ParamFunction, seed=42,
                        verbose=False):
    """
    """
    # Add restarting mechanism.
    metrics = Me.Metrics(input_channels=input_channels, seq_len=seq_len, kernel_sizes=kernel_size,
                         strides=strides, conv_paddings=conv_paddings, gpu=gpu, distance_metric=P.DistanceMetric,
                         gpu_device=gpu_device, prep_included=P.PreprocessingIncluded,
                         prep_excluded=P.PreprocessingExcluded, scale=P.Scale,
                         features_ordering=features_order, seed=seed, verbose=verbose,
                         data_fmt_class="MotionSenseDataset", window_overlap=P.Window_overlap)
    results = R.Results(resultDir=P.ResultsDir())
    sr = R.StopRestart(resultDir=P.ResultsDir())

    # Read data and compute metrics,
    # Baseline
    metric_epochs = 200
    metric_batch = 256
    def shaping(path):
        # Set the data as the same for all, the generated ones and the original ones such that we have the same
        # computation graph.
        data = D.MotionSenseDataset(path, window_overlap=P.Window_overlap)
        return data.__inverse_transform_conv__(sensor_tensor=data.sensor, phy=data.phy_data,
                                               act_tensor=data.activities, sens_tensor=data.sensitive,
                                               user_id_tensor=data.users_id, trials=data.trials, cpu_device=CPU_DEVICE)

    o_test = shaping(P.TestPath)
    if not sr.computed(epoch=0, alpha_=np.NaN, lambda_=np.NaN, Attribute="act") or \
        not sr.computed(epoch=0, alpha_=np.NaN, lambda_=np.NaN, Attribute="gender"):
        o_train = shaping(P.TrainPath)

        sp = metrics.sensitive_attribute(train_set=o_train, test_set=o_test,
                                         use_accuracy=False, act_name="act", drop=[], sens_name="gender", ms_act_name="act",
                                         ms_sens_name="sens", sklearn_data_process=None, use_phys=True, physNodes=3,
                                         phys_names=["height", "weight", "age"], ms_phys_name="phy", epoch=metric_epochs,
                                         batch_size=metric_batch, loss_fn=None)
        tp = metrics.task(train_set=o_train, test_set=o_test,
                          act_name="act", drop=[], sens_name="gender", ms_act_name="act", ms_sens_name="sens",
                          sklearn_data_process=None, use_phys=True, physNodes=3, phys_names=["height", "weight", "age"],
                          ms_phys_name="phy", epoch=metric_epochs, batch_size=metric_batch, loss_fn=None)
        di = metrics.distance(o_test, o_test)
        # Add and save results
        results.add_result(distance=di, s_acc=sp[0], s_ber=sp[1], t_acc=tp[0], t_ber=tp[1], sens_name="gender",
                           act_name="act", epoch=0, alpha_=np.NaN, lambda_=np.NaN)


    # Sanitization
    name = lambda n, e: "{}/{}_{}.csv".format(gen_path, n, addParamFn(e))
    for epoch in tqdm.tqdm(range(1, epoch+1)):
        if not sr.computed(epoch=epoch, alpha_=alpha_, lambda_=lambda_, Attribute="act") or \
            not sr.computed(epoch=epoch, alpha_=alpha_, lambda_=lambda_, Attribute="gender"):
            train = shaping(name(train_id, epoch))
            test = shaping(name(test_id, epoch))
            sp = metrics.sensitive_attribute(train_set=train, test_set=test,
                                             use_accuracy=False, act_name="act", drop=[], sens_name="gender", ms_act_name="act",
                                             ms_sens_name="sens", sklearn_data_process=None, use_phys=True, physNodes=3,
                                             phys_names=["height", "weight", "age"], ms_phys_name="phy", epoch=metric_epochs,
                                             batch_size=metric_batch, loss_fn=None, learning_rate=5e-4, weight_decay=0)
            tp = metrics.task(train_set=train, test_set=test,
                              act_name="act", drop=[], sens_name="gender", ms_act_name="act", ms_sens_name="sens",
                              sklearn_data_process=None, use_phys=True, physNodes=3, phys_names=["height", "weight", "age"],
                              ms_phys_name="phy", epoch=metric_epochs, batch_size=metric_batch, loss_fn=None,
                              learning_rate=5e-4, weight_decay=0)
            di = metrics.distance(o_test_set=o_test, test_set=test)
            # Add and save results
            results.add_result(distance=di, s_acc=sp[0], s_ber=sp[1], t_acc=tp[0], t_ber=tp[1], sens_name="gender",
                               act_name="act", epoch=epoch, alpha_=alpha_, lambda_=lambda_)


# sanitization_generation_phase()
sanitization_generation_metrics(feature_order=None, alpha_=P.Alpha, lambda_=P.Lambda, san_loss=P.SanLoss,
                                pred_loss=P.PredLoss, disc_loss=P.DiscLoss, max_epoch=P.Epoch, k_pred=P.KPred,
                                k_disc=P.KDisc, scale=P.Scale)
print("End")

# Compute metrics
