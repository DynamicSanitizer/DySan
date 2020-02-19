import warnings
import numpy as np
import pandas as pd
from Modules import Classifiers as C
from scipy.spatial import distance as Di


class Metrics:
    """
    Class to compute all metrics
    """
    def __init__(self, input_channels=6, seq_len=125, kernel_sizes=[5,5], strides=[1, 1],
                 conv_paddings=[0, 0], gpu=True, gpu_device="cuda:0", prep_included=None, distance_metric="euclidean",
                 prep_excluded=None, scale=0, features_ordering=None, seed=42, verbose=False,
                 data_fmt_class="MotionSenseDataset", **ds_kwargs):
        """
        Output size is automatically determined, based on the given training and test sets.
        :param ds_kwargs: other parameters to take into account when creating the data class. 
        """

        self.seed = seed
        self.verbose = verbose
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.output_size = None
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.conv_paddings = conv_paddings
        self.gpu = gpu
        self.gpu_device = gpu_device
        self.prep_included = prep_included
        self.prep_excluded = prep_excluded
        self.scale = scale
        self.features_ordering = features_ordering
        self.seed = seed
        self.verbose = verbose

        self.sklearn_data_process = None
        self.__target = None
        self.__sens = None
        self.__phys = None
        self.__drop = None

        self.__dis_m = getattr(Di, distance_metric)
        self.__distance_metric = distance_metric
        self.__split_fn = lambda d, t, drp: (d.drop([t, drp], axis=1), d[t])
        self.__ds_kwargs = ds_kwargs
        self.__data_fmt_class = data_fmt_class

    def __reset_sk_clfs__(self):
        self.__sk_clfs = C.Classifiers(seed=self.seed, verbose=self.verbose)

    def __reset_nn_task_clf__(self):
        self.__reset_sk_clfs__()
        self.__nn_clf = C.PredictorConv(model_class="PredictorConv", input_channels=self.input_channels,
                                        seq_len=self.seq_len, output_size=self.output_size,
                                        kernel_sizes=self.kernel_sizes, strides=self.strides,
                                        conv_paddings=self.conv_paddings, physNodes=self.physNodes, gpu=self.gpu,
                                        gpu_device=self.gpu_device, prep_included=self.prep_included,
                                        prep_excluded=self.prep_excluded, scale=self.scale,
                                        features_ordering=self.features_ordering, data_fmt_class=self.__data_fmt_class,
                                        **self.__ds_kwargs)
    def __reset_nn_sens_clf__(self):
        self.__reset_sk_clfs__()
        self.__nn_clf = C.PredictorConv(model_class="DiscriminatorConv", input_channels=self.input_channels,
                                        seq_len=self.seq_len, output_size=self.output_size,
                                        kernel_sizes=self.kernel_sizes, strides=self.strides,
                                        conv_paddings=self.conv_paddings, physNodes=self.physNodes, gpu=self.gpu,
                                        gpu_device=self.gpu_device, prep_included=self.prep_included,
                                        prep_excluded=self.prep_excluded, scale=self.scale,
                                        features_ordering=self.features_ordering, data_fmt_class=self.__data_fmt_class,
                                        **self.__ds_kwargs)

    def __fit__(self, data, target="sens", phys="phy", sens="sens", drop=None, epoch=200, batch_size=256,
                learning_rate=5e-4, weight_decay=0, loss_fn=None):
        """
        Apart from data, target, phy and sens, all other attributes are specific to the predictor created in models
        Each call update the parameters to use when calling predict
        """
        self.__target = target
        self.__phys = phys
        self.__sens = sens

        d = [target[0]]
        if drop is not None and len(drop[0]) != 0:
            d.extend(drop[0])
        if target[0] != sens[0]:
            d.append(sens[0])
        if not self.use_phys:
            d.append(phys[0])

        self.__drop = d

        data_sk = self.sklearn_data_process(data)
        self.__sk_clfs.fit(data_sk.drop(d, axis=1), data_sk[target[0]])

        # For nn, if the physical nodes drop is done automatically during model creation. By default,
        # nn do not use accuracy
        if loss_fn is not None:
            self.__nn_clf.fit(train_data=data, target=target[1], sens=sens[1], phys_clm=phys[1], epoch=epoch,
                              batch_size=batch_size, loss_fn=loss_fn, learning_rate=learning_rate,
                              weight_decay=weight_decay, verbose=self.verbose)
        else:
            self.__nn_clf.fit(train_data=data, target=target[1], sens=sens[1], phys_clm=phys[1], epoch=epoch,
                              batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay,
                              verbose=self.verbose)

    def __predict__(self, test_set):
        """
        return the predictions as well as the specific targets and the sensitive attribute
        :param test_set:
        :return:
        """
        data_sk = self.sklearn_data_process(test_set)
        p_sk = self.__sk_clfs.predict(data_sk.drop(self.__drop, axis=1))
        st = data_sk[self.__target[0]].values
        ss = data_sk[self.__sens[0]].values

        n, ns, nt = self.__nn_clf.predict(test_data=test_set, target=self.__target[1], sens=self.__sens[1], phys_clm=self.__phys[1])
        return (p_sk, st, ss), (n, nt, ns)

    @staticmethod
    def ber(pred, targ, sens):
        i = np.abs(pred != targ)
        d = pd.DataFrame.from_dict({"err": i, "sens": sens})
        return d.groupby(by="sens").mean().mean()["err"]

    @staticmethod
    def accuracy(pred, targ):
        return np.abs(pred == targ).mean()

    def sensitive_attribute(self, train_set, test_set, use_accuracy=False, act_name="act", drop=[], sens_name="gender",
                            ms_act_name="act", ms_sens_name="sens", sklearn_data_process=None, use_phys=True, physNodes=3,
                            phys_names=["height", "weight", "age"], ms_phys_name="phy", epoch=200, batch_size=256,
                            loss_fn=None, learning_rate=5e-4, weight_decay=0):
        """
        Train to predict the sensitive attribute and make the prediction and return the BER and the accuracy
        :param act_name: The name of the activity in the original dataset
        :param sens_name: The name of the sensitive attribute in the original dataset as it is when saved on disk
        :param ms_act_name: The name of the activity column when processed for the predictor or the discriminator
        :param ms_sens_name: Name of the sensitive attribute when the data has been processed
        :param sklearn_data_process: processing function for the sklearn classifiers. Must return a DataFrame

        set use_phys to False if drop contains phys attribute.
        Will be set for both the training and the test.
        :return:
        """

        use_accuracy = False # By default, nn does not use accuracy, so we do the same for sklearn.
        self.use_phys = use_phys
        self.physNodes = physNodes if use_phys else 0
        self.output_size = np.unique(train_set[sens_name].values).shape[0]
        self.sklearn_data_process = sklearn_data_process if sklearn_data_process is not None else lambda x:x
        self.__reset_nn_sens_clf__()

        if use_accuracy:
            self.__fit__(train_set, target=[sens_name, ms_sens_name], phys=[phys_names, ms_phys_name],
                         sens=[sens_name, ms_sens_name], drop=drop, epoch=epoch, batch_size=batch_size,
                         loss_fn=loss_fn, learning_rate=learning_rate, weight_decay=weight_decay)
        else:
            d = [act_name]
            d.extend(drop)
            self.__fit__(train_set, target=[sens_name, ms_sens_name], phys=[phys_names, ms_phys_name],
                         sens=[sens_name, ms_sens_name], drop=[d, ms_act_name], epoch=epoch, batch_size=batch_size,
                         loss_fn=loss_fn, learning_rate=learning_rate, weight_decay=weight_decay)

        sk_pr, n_pr = self.__predict__(test_set)
        a = {}
        b = {}
        for c, p in sk_pr[0].items():
            a.update({c: self.accuracy(p, sk_pr[1])})
            b.update({c: self.ber(pred=p, targ=sk_pr[1], sens=sk_pr[2])})

        for c, p in n_pr[0].items():
            a.update({c: self.accuracy(p, n_pr[1])})
            b.update({c: self.ber(pred=p, targ=n_pr[1], sens=n_pr[2])})
        return a, b

    def task(self, train_set, test_set, act_name="act", drop=[], sens_name="gender", ms_act_name="act",
             ms_sens_name="sens", sklearn_data_process=None, use_phys=True, physNodes=3,
             phys_names=["height", "weight", "age"], ms_phys_name="phy", epoch=200, batch_size=256,
             loss_fn=None, learning_rate=5e-4, weight_decay=0):
        """
        Train for a specific task prediction
        """
        if train_set[act_name].value_counts().shape[0] > 1:
            self.use_phys = use_phys
            self.physNodes = physNodes if use_phys else 0
            self.output_size = np.unique(train_set[act_name].values).shape[0]
            self.sklearn_data_process = sklearn_data_process if sklearn_data_process is not None else lambda x:x
            self.__reset_nn_task_clf__()

            # train_set.drop([sens_name], axis=1,)
            # test_set.drop([sens_name], axis=1, )
            d = [sens_name]
            d.extend(drop)
            self.__fit__(train_set, target=[act_name, ms_act_name], phys=[phys_names, ms_phys_name],
                         sens=[sens_name, ms_sens_name], drop=[d, ms_sens_name], epoch=epoch, batch_size=batch_size,
                         loss_fn=loss_fn, learning_rate=learning_rate, weight_decay=weight_decay)
            # self.fit(train_set, target=[target, phys=phys, drop=drop, epoch=epoch, batch_size=batch_size, loss_fn=loss_fn,
            #          verbose=verbose)

            sk_pr, n_pr = self.__predict__(test_set)
            a = {}
            b = {}
            for c, p in sk_pr[0].items():
                a.update({c: self.accuracy(p, sk_pr[1])})
                b.update({c: self.ber(pred=p, targ=sk_pr[1], sens=sk_pr[2])})

            for c, p in n_pr[0].items():
                a.update({c: self.accuracy(p, n_pr[1])})
                b.update({c: self.ber(pred=p, targ=n_pr[1], sens=n_pr[2])})
        else:
            # Cannot train the models set everything to Not a Number.
            a = self.__sk_clfs.return_np_nan()
            a.update(self.__nn_clf.return_np_nan())
            b = self.__sk_clfs.return_np_nan()
            b.update(self.__nn_clf.return_np_nan())

        return a, b

    def distance(self, o_test_set, test_set):
        """
        Compute the distance between the original test_set and the sanitized one
        :param o_test_set: the original test set
        :param test_set: the sanitized test set
        :return: the computed distance
        """
        r = lambda d: d.values.reshape(-1) if isinstance(d, pd.DataFrame) else d.reshape(-1)
        s = lambda s: pd.read_csv(s) if isinstance(s, str) else s
        # if isinstance(o_test_set, pd.DataFrame):
        #     o_test_set = o_test_set.values
        # if isinstance(test_set, pd.DataFrame):
        #     test_set = test_set.values
        o_test_set = r(s(o_test_set))
        test_set = r(s(test_set))
        d = self.__dis_m(o_test_set, test_set)
        return {self.__distance_metric: d}
