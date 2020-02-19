import copy
import torch
import numpy as np
import pandas as pd
from torch.utils import data
from sklearn.preprocessing import MinMaxScaler

class Preprocessing:
    """
    Class to handle all the preprocessing steps before passing the dataset to pytorch
    """

    def __init__(self, csv, numeric_as_categoric_max_thr=5, scale=0, prep_excluded=None, prep_included=None):
        """

        :param csv: the path to the dataset
        :param numeric_as_categoric_max_thr: Maxiimum number of values for a numeric column to be considered as
        categoric
        :param scale: the lower bound of the feature scaling. Either 0 or -1
        :param prep_excluded: Columns to exclude from the preprocessing
        :param prep_included: Columns to include from the preprocessing. This overwrite the prep_excluded
        """
        if isinstance(csv , str):
            self.df = pd.read_csv(csv)
            if 'Unnamed: 0' in self.df.columns:
                self.df = self.df.drop(['Unnamed: 0'], axis = 1)
        elif isinstance(csv, pd.DataFrame):
            self.df = csv.copy(True)
        else:
            raise ValueError("Invalid input type: {}".format(csv))

        self.cols_order = self.df.columns
        self.prep_excluded = prep_excluded
        self.prep_included = prep_included
        if prep_included is not None:
            self.prep_excluded = list(set(self.cols_order) - set(prep_included))
        self.features_ordering = None
        self.categorical_was_numeric = {}
        self.num_as_cat = numeric_as_categoric_max_thr
        self.scale = scale
        self.scaler = MinMaxScaler(feature_range=(scale, 1))

    def __find_categorical__(self):
        """
        List int and float columns
        :return: The list of num column
        """
        cat_clm = []
        num_clm = []
        for c in self.df.select_dtypes(include=["int", 'float', 'double']).columns:
            if self.df[c].value_counts().shape[0] <= self.num_as_cat:
                self.categorical_was_numeric.update({c: self.df[c].dtype})
                cat_clm.append(c)
            else:
                num_clm.append(c)
        cat_clm.extend(self.df.select_dtypes(exclude=["int", 'float', 'double']).columns.tolist())
        self.cat_clm = cat_clm
        self.num_clm = num_clm
        self.int_cls = self.df.select_dtypes(include=["int"]).columns.tolist()

    def __from_dummies__(self, prefix_sep='='):
        """
        Convert encoded columns into original ones
        """
        data = self.df
        categories = self.cat_clm
        cat_was_num = self.categorical_was_numeric
        out = data.copy()
        for l in categories:
            cols = data.filter(regex="^{}{}".format(l, prefix_sep), axis=1).columns
            labs = [cols[i].split(prefix_sep)[-1] for i in range(cols.shape[0])]
            out[l] = pd.Categorical(np.array(labs)[np.argmax(data[cols].values, axis=1)])
            out.drop(cols, axis=1, inplace=True)
            if l in cat_was_num.keys():
                out[l] = out[l].astype(cat_was_num[l])

        self.df = out

    def __squash_in_range__(self):
        """
        Squash df values between min and max from scaler
        """
        for c in self.num_clm:
            i = self.df.columns.get_loc(c)
            self.df.loc[self.df[c] > self.scaler.data_max_[i], c] = self.scaler.data_max_[i]
            self.df.loc[self.df[c] < self.scaler.data_min_[i], c] = self.scaler.data_min_[i]

    def set_features_ordering(self, features_ordering):
        self.features_ordering = features_ordering

    def set_features_params(self, features_ordering, categorical_was_numeric, num_clm, cat_clm):
        """
        Set the original feature ordering, and the original type of numeric columns that have been considered
        as categorical because of they had a limited number of values
        """
        self.features_ordering = features_ordering
        self.categorical_was_numeric = categorical_was_numeric
        self.num_clm = num_clm
        self.cat_clm = cat_clm

    def __features_formatting__(self):
        """
        Add missing columns to have the same dataset structure
        scale should be the lower scale as given i
        """
        if self.features_ordering is not None:
            for c in self.features_ordering:
                if c not in self.df.columns and c not in self.prep_excluded:
                    self.df[c] = self.scale

            self.df = self.df[self.features_ordering]

    def __round_integers__(self):
        """
        Round the columns that where of type integer to begin with
        """
        for c in self.int_cls:
            self.df[c] = self.df[c].round().astype("int")

    def fit_transform(self, prefix_sep='='):
        """
        Apply all transformation
        """
        excluded = pd.DataFrame()
        # if self.prep_included is not None and len(self.prep_included) > 0:
        #     excluded = self.df[self.prep_included] # Included
        #     self.df = self.df.drop(self.prep_included, axis=1) # Excluded
        #     excluded, self.df = self.df, excluded # Inverting both
        # else:
        if self.prep_excluded is not None:
             excluded = self.df[self.prep_excluded]
             self.df.drop(self.prep_excluded, axis=1, inplace=True)

        self.__find_categorical__()
        self.df = pd.get_dummies(self.df, columns=self.cat_clm, prefix_sep=prefix_sep)

        # Scale the data, then add missing columns and set a fixed order, then add the excluded columns. Excluded is at
        # the end of the processing as we are not supposed to touch them.
        # columns
        # Scaler contains all columns except the excluded ones
        # self.df.iloc[:, :] = self.scaler.fit_transform(self.df.values)
        self.df = pd.DataFrame(self.scaler.fit_transform(self.df.values), columns=self.df.columns)
        # Complete missing columns, and give a standard column order.
        self.__features_formatting__()

        self.df = pd.concat([self.df, excluded], axis=1)[self.cols_order]

        # Complete missing columns
        # self.__features_formatting__()
        # Scaler contains all columns
        # self.df.iloc[:, :] = self.scaler.fit_transform(self.df.values)
        # self.df = pd.DataFrame(self.scaler.fit_transform(self.df.values), columns=self.df.columns)
        # self.df = pd.concat([self.df, excluded], axis=1)


    def inverse_transform(self):
        """
        Recover the original data
        """
        excluded = pd.DataFrame()
        # if self.prep_included is not None and len(self.prep_included) > 0:
        #     excluded = self.df[self.prep_included] # Included
        #     self.df = self.df.drop(self.prep_included, axis=1) # Excluded
        #     excluded, self.df = self.df, excluded # Inverting both
        # else:
        if self.prep_excluded is not None:
            excluded = self.df[self.prep_excluded]
            self.df.drop(self.prep_excluded, axis=1, inplace=True)
        self.df = pd.DataFrame(self.scaler.inverse_transform(self.df.values), columns=self.df.columns)
        # Scaler contains all columns
        self.__squash_in_range__()
        self.__round_integers__()
        self.__from_dummies__()
        self.df = pd.concat([self.df, excluded], axis=1)[self.cols_order]

    def copy(self, deepcopy=False):
        if deepcopy:
            return copy.deepcopy(self)
        return copy.copy(self)


class GeneralDataset(data.Dataset):

    def __init__(self, csv, target_feature, xTensor=torch.FloatTensor, yTensor=torch.LongTensor, transform=None,
                 to_drop=None):
        """
        Do all the heavy data processing here.
        :param csv: the filename with data or the data as dataFrame or an instance of Preprocessing class
        :param target_feature: the target feature name
        :param xTensor: data tensor type
        :param yTensor: target tensor type
        :param transform: transformations from pytorch to apply
        :param to_drop: list of features to drop
        """
        if isinstance(csv, Preprocessing):
            df = csv.df
        elif isinstance(csv, pd.DataFrame):
            df = csv
        else:
            df = pd.read_csv(csv)
            if 'Unnamed: 0' in self.df.columns:
                self.df = self.df.drop(['Unnamed: 0'], axis = 1)
        self.length = len(df)
        drop = [target_feature]
        if to_drop is not None:
            drop.extend(to_drop)
        self.y = yTensor(df[target_feature].values)
        self.x = xTensor(df.drop(drop, axis=1).values)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sample = {'x': self.x[index], 'y': self.y[index]}
        return sample


class MotionSenseDataset(data.Dataset):
    """
    Class for the motion sense dataset
    """
    def __init__(self, csv, act_name="act", uid_name="id", sens_name="gender", time_window=2.5, window_overlap=1.25,
                 sample_rate=50, sensorTensor=torch.FloatTensor, actTensor=torch.LongTensor, idTensor=torch.LongTensor,
                 transform=None, to_drop=None, trial_feature="trial", physiologic_feat=["weight", "height", "age"],
                 seq_padding=None, rnn=False):
        """
        Initialisation. Doing all the heavy processing here. The trial column must be present in order to group data.
        Cannot group the same activities of two different trials together. If no trial, then we assume each activity is
        performed without collisions (once)
        :param csv: The data to use either a dataframe or an instance of Preprocessing class
        :param act_name: the activity column name
        :param sens_name: Sensitive attribute name
        :param uid_name: the user id column name
        :param time_window: the windows size to consider
        :param window_overlap: The overlapping of data sequences to consider, how much overlap each timestep
        will have with the previous one
        :param sample_rate: the data collection sample rate
        :param sensorTensor: the sensor data tensor
        :param actTensor: the activity data tensor
        :param idTensor: the user id tensor
        :param transform: other transformation to perform
        :param to_drop: list columns names to drop
        :param trial_feature: The feature name of trial data
        :param physiologic_feat: The physiologic data, which are not a sequence
        :param seq_padding: the padding to add to a sequence in order to make it have the same length as any other
        :param rnn: Make the sequence as for the rnn format, instead of the cnn
        """

        if isinstance(csv, Preprocessing):
            df = csv.df
        elif isinstance(csv, pd.DataFrame):
            df = csv
        else:
            df = pd.read_csv(csv)
            if 'Unnamed: 0' in self.df.columns:
                self.df = self.df.drop(['Unnamed: 0'], axis = 1)
        self.cols_recv_order = df.columns.tolist()

        drop = [act_name, uid_name, sens_name]
        trial = False
        self.trial_name = None
        if trial_feature in df.columns:
            trial = True
            self.trial_name = trial_feature
            drop.append(trial_feature)
        if to_drop is not None:
            drop.extend(to_drop)

        sensor_data = []
        # act_id_sens = df[act_feature, id_feature, sensitive]

        window = int(sample_rate * time_window)
        self.window = window
        assert (window_overlap < window), "The overlap MUST be less than the time window"
        overlap = int(sample_rate * window_overlap)
        self.overlap = overlap
        c_targets = [uid_name, act_name, sens_name]
        id_act_sens = pd.DataFrame(columns=c_targets)
        # Seems like if the data ttype if int, it will not modify the type object whereas if it is float, the type object will be casted to float automatically when adding new values (concat)
        id_act_sens = id_act_sens.astype(dict(zip(c_targets, df[c_targets].dtypes.values.tolist())))
        phy_data = pd.DataFrame(columns=physiologic_feat)

        # Get the upper limit that differentiate each users. Will be used for shuffling the dataset
        user_map = []
        trials = []
        self.sensor_features_name = None
        self.physiologic_feat = physiologic_feat
        self.uid_name = uid_name
        self.act_name = act_name
        self.sens_name = sens_name

        def _by_uid_sequences_(d, id_feature, id_act_sens, c_targets, phy_data, physiologic_feat, sensor_data,
                              window, overlap, seq_padding, user_map):

            def add_other(id_act_sens,phy_data, di, c_targets=c_targets, physiologic_feat=physiologic_feat):
                # Add single point
                id_act_sens = pd.concat([id_act_sens, di.iloc[:1][c_targets]], axis=0)
                phy_data = pd.concat([phy_data, di.iloc[:1][physiologic_feat]], axis=0)
                return id_act_sens, phy_data

            # Group by user ids
            for i in d[id_feature].value_counts().index:
                # Create block of sequence. Each sequence correspond to a timestep define by
                # sample_rate / time_windows
                # Same user, hence same id and same gender. Same trial hence same activity
                di = d[d[id_feature] == i]

                # di should contains only the sensor values
                di2 = di.drop(drop, axis=1).drop(physiologic_feat, axis=1)
                self.sensor_features_name = di2.columns.tolist()
                b = 0
                while b < di.shape[0]:
                    # Each loop is a single timestep. The timestep will be similar to the batch id
                    # concatenate data at  each time step
                    v = di2.iloc[b:b+window].T.values
                    # Making all sequence with the same length
                    if v.shape[1] < window:
                        if seq_padding is not None:
                            o = np.ones((v.shape[0], window - v.shape[1])) * seq_padding
                            vo = np.concatenate((v, o), axis=1)
                            # Adding values
                            # Drop sequence if len < window we already have enough data
                            sensor_data.append(vo)
                            id_act_sens, phy_data = add_other(id_act_sens, phy_data, di, c_targets, physiologic_feat)
                    else:
                        sensor_data.append(v)
                        id_act_sens, phy_data = add_other(id_act_sens, phy_data, di, c_targets, physiologic_feat)

                    b += window - overlap
                user_map.append(phy_data.shape[0])

            return id_act_sens, phy_data

        if trial:
            # Each trial correspond to a single activity, for all user. Group by users
            for t in df[trial_feature].value_counts().index:
                d = df[df[trial_feature] == t]
                id_act_sens, phy_data = _by_uid_sequences_(d, uid_name, id_act_sens, c_targets, phy_data,
                                                           physiologic_feat, sensor_data, window, overlap, seq_padding,
                                                           user_map)
                trials.extend(np.repeat(t, phy_data.shape[0] - len(trials)).tolist())
        else:
            # Group by activity then by users.
            trials = None
            for a in df[act_name].value_counts().index:
                # Single activity just as with trial above.
                d = df[df[act_name] == a]
                id_act_sens, phy_data = _by_uid_sequences_(d, uid_name, id_act_sens, c_targets, phy_data,
                                                           physiologic_feat, sensor_data, window, overlap, seq_padding,
                                                           user_map)


        self.sensor = sensorTensor(np.array(sensor_data))
        # Physiologic data can be concatenated at the dense layres
        self.phy_data = sensorTensor(phy_data.values)
        self.activities = actTensor(id_act_sens[act_name].values)
        self.sensitive = actTensor(id_act_sens[sens_name].values)
        self.users_id = idTensor(id_act_sens[uid_name].values)
        self.user_map = user_map
        self.trials = trials
        self.seq_len = window
        self.input_channels = self.sensor.shape[1] # Format is : timestep, num_features (channels), seq_len or window
        self.rec_infos = id_act_sens
        self.rec_infos[self.trial_name] = trials
        # size.
        assert(self.sensor.shape[0] == self.users_id.shape[0] == self.activities.shape[0]
               == self.sensitive.shape[0] == self.phy_data.shape[0]), "Data creation have a serious mistake." \
                                                                      " Need review"
        self.length = self.sensitive.shape[0]
        self.transform = transform

        if not rnn:
            # Shuffle sequences, we know that the matrices are ordered by user id. Randomize such ids, but do not
            # modified each user timestep order.
            # To much complicated. Maybe shuffle only by group of user ids. Or maybe define a new randomsampler ?
            pass

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        sample = {
            "sensor": self.sensor[index],
            "phy": self.phy_data[index],
            "sens": self.sensitive[index],
            "act": self.activities[index],
            "uid": self.users_id[index]
        }
        return sample

    def __inverse_transform_conv_before__(self, sensor_tensor, phy, act_tensor, sens_tensor, user_id_tensor, trials=None,
                                   cpu_device="cpu"):
        """
        Recompose the original data as dataframe, feature unscalling and bounds are not realised here.
        Only the form is transformed. Use this is the data is transformed for usage in a conv network
        :return:
        """
        assert len(sensor_tensor.shape) == 3, "Sensor shape must have 3 dimensions"
        sensor_features = sensor_tensor.shape[1]
        assert sensor_features == len(self.sensor_features_name), "Number of sensor features and features name does not match"

        # data = pd.DataFrame()
        # Faster to work with matrices. See data2.
        # for s, p, ss, a, u, t in zip(sensor_tensor.to(device), phy.to(device), sens_tensor.view(-1).to(device),
        #                              act_tensor.view(-1).to(device), user_id_tensor.view(-1).to(device), trials):
        #     s = s.data.numpy()
        #     d = dict(zip(self.sensor_features_name, s.tolist()))
        #
        #     p = np.repeat(p.data.numpy().reshape(1, -1), s.shape[1], axis=0).transpose().tolist()
        #     d.update(dict(zip(self.physiologic_feat, p)))
        #     ss = np.repeat(ss.data.numpy(), s.shape[1]).tolist()
        #     d.update({self.sens_name: ss})
        #     a = np.repeat(a.data.numpy(), s.shape[1]).tolist()
        #     d.update({self.act_name: a})
        #     u = np.repeat(u.data.numpy(), s.shape[1]).tolist()
        #     d.update({self.uid_name: u})
        #     if trials is not None:
        #         d.update({self.trial_name: np.repeat(t, s.shape[1]).tolist()})
        #
        #     data = pd.concat([data, pd.DataFrame.from_dict(d)], axis=0)

        sensor_tensor = sensor_tensor.to(cpu_device)
        phy = phy.to(cpu_device)
        sens_tensor = sens_tensor.view(-1, 1).to(cpu_device)
        act_tensor = act_tensor.view(-1, 1).to(cpu_device)
        user_id_tensor = user_id_tensor.view(-1, 1).to(cpu_device)

        f = {}
        sensor = []
        def matrix_to_vector(mat):
            if self.overlap == 0:
                return mat.reshape(-1)
            else:
                d = self.window - self.overlap
                size = mat.shape[1] + d*(mat.shape[0] - 1)
                sequence = np.zeros((1, size))
                sequence[0, 0:self.window] = mat[0, :]
                counter = np.zeros((1, size))
                counter[0, 0:self.window] = 1

                for i in range(1, mat.shape[0]):
                    # Create temp windows
                    t = np.zeros((1, size))
                    ct = np.zeros((1, size))
                    # Shift elements
                    t[0, d:d+self.window] = mat[i, :]
                    ct[0, d:d + self.window] = 1
                    # Summation
                    sequence += t
                    counter += ct
                    d += self.window - self.overlap
                # Divide and reshape
                sequence /= counter
                return sequence.reshape(-1)

                # Big matrix, need a ot of memory
                # z = np.ones((mat.shape[0], size)) * np.NaN
                # b = 0
                # s = mat.shape[1]
                # for i in range(mat.shape[0]):
                #     z[i, b:b + s] = mat[i, :]
                #     b += s - self.overlap
                # return np.nanmean(z, axis=0)

        for f_index in range(sensor_features):
            sensor.append(matrix_to_vector(sensor_tensor[:, f_index, :].data.numpy()).tolist())
        f.update(dict(zip(self.sensor_features_name, sensor)))

        expand = lambda v: v.view(-1, 1).repeat(1, sensor_tensor.shape[2]).view(-1).data.numpy().tolist()
        phys = []
        for c in range(phy.shape[1]):
            phys.append(expand(phy[:, c]))
        f.update(dict(zip(self.physiologic_feat, phys)))

        f.update({self.sens_name: expand(sens_tensor)})
        f.update({self.act_name: expand(act_tensor)})
        f.update({self.uid_name: expand(user_id_tensor)})
        if trials is not None:
            f.update({self.trial_name: expand(torch.FloatTensor(trials))})

        data2 = pd.DataFrame.from_dict(f)

        data2 = data2[self.cols_recv_order]
        # Set a column order that will be applied to all datasets.
        if trials is not None:
            data2 = data2.sort_values(by=[self.trial_name, self.act_name, self.uid_name], ascending=False)
        else:
            data2 = data2.sort_values(by=[self.act_name, self.uid_name], ascending=False)
        return data2


    def copy(self, deepcopy=False):
        if deepcopy:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def __inverse_transform_conv__(self, sensor_tensor, phy, act_tensor, sens_tensor, user_id_tensor, trials=None,
                                   cpu_device="cpu"):
        """
        Recompose the original data as dataframe, feature unscalling and bounds are not realised here.
        Only the form is transformed. Use this is the data is transformed for usage in a conv network
        :return:
        """
        assert len(sensor_tensor.shape) == 3, "Sensor shape must have 3 dimensions"
        sensor_features = sensor_tensor.shape[1]
        assert sensor_features == len(self.sensor_features_name), "Number of sensor features and features name does not match"

        sensor_tensor = sensor_tensor.to(cpu_device)
        phy = phy.to(cpu_device)
        sens_tensor = sens_tensor.view(-1, 1).to(cpu_device)
        act_tensor = act_tensor.view(-1, 1).to(cpu_device)
        user_id_tensor = user_id_tensor.view(-1, 1).to(cpu_device)

        f = {}
        sensor = []
        def matrix_to_vector(mat):
            if self.overlap == 0:
                return mat.reshape(-1)
            else:
                d = self.window - self.overlap
                size = mat.shape[1] + d*(mat.shape[0] - 1)
                sequence = np.zeros((1, size))
                sequence[0, 0:self.window] = mat[0, :]
                counter = np.zeros((1, size))
                counter[0, 0:self.window] = 1

                for i in range(1, mat.shape[0]):
                    # Create temp windows
                    t = np.zeros((1, size))
                    ct = np.zeros((1, size))
                    # Shift elements
                    t[0, d:d+self.window] = mat[i, :]
                    ct[0, d:d + self.window] = 1
                    # Summation
                    sequence += t
                    counter += ct
                    d += self.window - self.overlap
                # Divide and reshape
                sequence /= counter
                return sequence.reshape(-1)


        for f_index in range(sensor_features):
            sensor.append([])
        p = self.physiologic_feat if isinstance(self.physiologic_feat, list) else [self.physiologic_feat]
        f.update({"p": None})

        f.update({self.sens_name: []})
        f.update({self.act_name: []})
        f.update({self.uid_name: []})
        col_n = self.act_name
        if trials is not None:
            f.update({self.trial_name: []})
            col_n = self.trial_name
        extend = lambda n, v, r: f[n].extend(v.repeat(r).data.numpy().tolist())
        # Sequences are created based on activities and user ids. No overlap is made between user data
        # We know that if there are more than 1 user, then the user_id vector will be [111, 222, 111, 222, 111, ...] where the first occurence of each user id represent the first activity,
        # the second occurence the second activity, etc... we reconstruct knowing that and where there is an overlap with different values, we take the mode which represent the values mostly
        # used for  each activity.
        for trial_or_act in self.rec_infos[col_n].value_counts().index:
            for id_ in self.rec_infos[self.uid_name].value_counts().index:
                mask = (self.rec_infos[self.uid_name] == id_) & (self.rec_infos[col_n] == trial_or_act)
                mask = torch.ByteTensor(mask.astype(int).values.tolist())
                # Get specific data
                sst = sensor_tensor[mask]
                for f_index in range(sensor_features):
                    s = matrix_to_vector(sst[:, f_index, :].data.numpy())
                    sensor[f_index].extend(s.tolist())
                # For other data, we select the mode. We suppose that we should haev a single value that is repeated because we restrict the data here to a single user and a single activity.
                # Single user measn single phy data and single id. Hence if during the sanitization we obtain several values, the mode should be used as it will represent the most frequently
                # produced by the sanitization procedure. We believe that when the whole system will converge, we might end up with data very close to each other, hence taking the mode
                # will make more sense than the mean or anything else. The same reasonning is applied on the activities here.
                act_mode, _ = act_tensor[mask].mode(dim=0)
                uid_mode, _ = user_id_tensor[mask].mode(dim=0) # kind of useless since it is already saved in rec_infos
                sens_mode, _ = sens_tensor[mask].mode(dim=0)
                phys_modes, _ = phy[mask].mode(dim=0)
                # We repeat the mode
                extend(self.act_name, act_mode, s.shape[0])
                extend(self.uid_name, uid_mode, s.shape[0])
                extend(self.sens_name, sens_mode, s.shape[0])

                if f["p"] is None:
                    f["p"] = phys_modes.view(1, -1).repeat(s.shape[0], 1).data.numpy()
                else:
                    f["p"] = np.concatenate([f["p"], phys_modes.view(1, -1).repeat(s.shape[0], 1).data.numpy()], 0)
                if trials is not None:
                    f[self.trial_name].extend(np.array([trial_or_act]).repeat(s.shape[0]).tolist())
                # f[self.act_name].extend(activities.repeat(s.shape[0]).data.numpy().tolist())


        f.update(dict(zip(self.sensor_features_name, sensor)))
        f.update(dict(zip(self.physiologic_feat, f["p"].T.tolist())))
        del f["p"]
        data2 = pd.DataFrame.from_dict(f)

        data2 = data2[self.cols_recv_order]
        # Set a column order that will be applied to all datasets.
        if trials is not None:
            data2 = data2.sort_values(by=[self.trial_name, self.act_name, self.uid_name], ascending=False)
        else:
            data2 = data2.sort_values(by=[self.act_name, self.uid_name], ascending=False)
        return data2


# To shuffle batch, use numpy and torch to define the new index order and shuflle all of the 5 data in the same way

