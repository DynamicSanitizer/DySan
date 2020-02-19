import tqdm
import numpy as np
import pandas as pd

from sklearn import ensemble, neural_network, svm, gaussian_process, linear_model, tree
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning

from torch.utils import data

from Modules import Models as M
from Modules import Datasets as D
from Modules import CustomLosses as Cl

class PredictorConv:
    """
    Class to train the predictor from Models in an user abstract fashioin
    """
    def __init__(self, model_class="PredictorConv", input_channels=6, seq_len=125, output_size=4, kernel_sizes=[5,5], strides=[1, 1],
                 conv_paddings=[0, 0], physNodes=0, gpu=True, gpu_device="cuda:0", prep_included=None,
                 data_fmt_class="MotionSenseDataset", prep_excluded=None, scale=0, features_ordering=None, **ds_kwargs):
        """

        :param input_channels:
        :param seq_len:
        :param output_size:
        :param kernel_sizes:
        :param strides:
        :param conv_paddings:
        :param physNodes:
        :param cpu:
        :param cpu_device:
        :param data_fmt_class: The data class to use to format the dataset such that it can be used by the model.
        Data class called after the preprocessing step, and before the dataloader step
        :param ds_kwargs: other arguments for the data_fmt_class chosen.
        """

        self.predictor = getattr(M, model_class)(input_channels=input_channels, seq_len=seq_len, output_size=output_size,
                                         kernel_sizes=kernel_sizes, strides=strides, conv_paddings=conv_paddings,
                                         physNodes=physNodes)

        self.prep_included = prep_included
        self.prep_excluded = prep_excluded
        self.scale = scale
        self.gpu = gpu
        self.features_ordering = features_ordering
        self.d_class  = getattr(D, data_fmt_class)
        self.d_class_kwargs = ds_kwargs

        self.device = "cpu"
        if gpu:
            # Send model on GPU or CPU
            self.device = gpu_device
        self.predictor.to(self.device)

    def __fit__(self, tr_data, sens, target, phys_clm, optim, loss_fn):
        # print("Epoch: {}".format(i))
        # set model to train and initialize aggregation variables
        self.predictor.train()
        # for each batch
        # get the optimizer (allows for changing learning rates)
        for sample in tr_data:
            # put each of the batch objects on the device
            x = sample['sensor'].to(self.device)
            p = sample[phys_clm].to(self.device)
            s = sample[sens].to(self.device)
            # u = sample["uid"].to(device)
            # y = sample['act'].unsqueeze(1).to(device)
            y = sample[target].to(self.device)
            yp = self.predictor(x, p)
            l = loss_fn(yp, y, s)
            optim.zero_grad()
            l.backward()
            optim.step()


    def fit(self, train_data, target="sens", sens="sens", phys_clm="phy", epoch=200, batch_size=256, learning_rate=5e-4,
            weight_decay=0, loss_fn=Cl.BalancedErrorRateLoss(1 / 2), verbose=False):
        """
        :param train_data: data to use for training. Must be an instance of pandas DataFrame
        :param target: the name of the target column
        :param phys_clm: the name of the physical columns
        """
        assert isinstance(train_data, pd.DataFrame), "The given data must be an instance of pandas DataFrame"
        assert isinstance(target, str), "Target must be the column name"
        assert isinstance(phys_clm, str), "phys_clm must be a string"
        # assert callable(loss_fn), "{} is not callable".format(loss_fn)

        tr_data = D.Preprocessing(train_data, prep_excluded=self.prep_excluded, scale=self.scale,
                                  prep_included=self.prep_included)
        tr_data.set_features_ordering(self.features_ordering)
        tr_data.fit_transform()
        tr_data = self.d_class(tr_data, **self.d_class_kwargs)
        tr_data = data.DataLoader(tr_data, batch_size=batch_size, shuffle=True, num_workers=4)

        optim = M.get_optimizer(self.predictor, lr=learning_rate, wd=weight_decay)
        if hasattr(loss_fn, "device"):
            loss_fn.device = self.device

        if verbose:
            print("Training predictor")
            for i in tqdm.tqdm(range(epoch)):
                self.__fit__(tr_data=tr_data, sens=sens, target=target, phys_clm=phys_clm, optim=optim, loss_fn=loss_fn)
        else:
            for i in range(epoch):
                self.__fit__(tr_data=tr_data, sens=sens, target=target, phys_clm=phys_clm, optim=optim, loss_fn=loss_fn)
        self.predictor.train(False)


    def predict(self, test_data, target="sens", sens="sens", phys_clm="phy"):
        """
        Return the prediction, as well as the target groundtruth and the sensitive groundtruth, since some data
         processing has been done
        :param test_data:
        :param target:
        :param sens:
        :param phys_clm:
        :return:
        """
        assert isinstance(test_data, pd.DataFrame), "The given data must be an instance of pandas DataFrame"
        assert isinstance(target, str), "Target must be the column name"
        assert isinstance(phys_clm, str), "phys_clm must be a string"
        ts_data = D.Preprocessing(test_data, prep_excluded=self.prep_excluded, scale=self.scale,
                                  prep_included=self.prep_included)
        ts_data.set_features_ordering(self.features_ordering)
        ts_data.fit_transform()
        ts_data = self.d_class(ts_data, **self.d_class_kwargs)
        ts_data = data.DataLoader(ts_data, batch_size=ts_data.sensor.shape[0], shuffle=False, num_workers=4)
        # Single loop since the batch size correspond to the test set size
        for sample in ts_data:
            # put each of the batch objects on the device
            x = sample['sensor'].to(self.device)
            p = sample[phys_clm].to(self.device)
            s = sample[sens]
            t = sample[target]
            # u = sample["uid"].to(device)
            # y = sample['act'].unsqueeze(1).to(device)
            yp = self.predictor(x, p).argmax(1)
        try:
            return {self.__class__.__name__: yp.data.numpy()}, s.data.numpy(), t.data.numpy()
        except TypeError:
            return {self.__class__.__name__: yp.cpu().data.numpy()}, s.data.numpy(), t.data.numpy()

    def return_np_nan(self):
        return {self.__class__.__name__: np.NaN}


class Classifiers:
    """
    Class list of all classifier used to make prediction.
    Simple wrapper for all classifiers used.
    """

    def __init__(self, addition=None, addition_names=None, default=True, seed=42, verbose=False):
        """
        :param addition: New classifiers to add in the list for making prediction
        :param addition_names: Names of the added classifiers
        :param default: Use the default list of classifiers. If set to False,
         addition must be provided
        """
        assert(default or ((addition is not None) and (addition_names is not None)), "No classifiers provided. At least set "
                                                                             "default to True")
        self.listClfs = {}
        if default:
            tc = [
                ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed),
                neural_network.MLPClassifier(random_state=seed),
                # svm.SVC(class_weight="balanced", gamma="scale"),
                tree.DecisionTreeClassifier(),
                ensemble.RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed),
                linear_model.LogisticRegression(class_weight='balanced', solver='liblinear')

            ]
            for c in tc:
                self.listClfs.update({c.__str__().split("(")[0]: c})
        if (addition_names is not None) and (addition is None):
            self.listClfs.update(dict(zip(addition_names, addition)))

        self.v = verbose
        self.verbose = lambda s: print(s) if self.v else None
        self.verbose("List of Classifiers:")
        self.verbose(self.listClfs)
        self.fitted = False

    @ignore_warnings(category=ConvergenceWarning)
    @ignore_warnings(category=DataConversionWarning)
    def fit(self, *args, **kwargs):
        for clf in self.listClfs.items():
            self.verbose("--- Training {}\n".format(clf[0]))
            clf[1].fit(*args, **kwargs)

    @ignore_warnings(category=DataConversionWarning)
    def predict(self, *args, **kwargs):
        p = {}
        for clf in self.listClfs.items():
            self.verbose("--- Predicting {}\n".format(clf[0]))
            p.update({clf[0]: clf[1].predict(*args, **kwargs)})

        return p

    def return_np_nan(self):
        p = {}
        for clf in self.listClfs.items():
            self.verbose("--- Setting NaN to {}\n".format(clf[0]))
            p.update({clf[0]: np.NaN})
        return p
