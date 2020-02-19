import numpy as np
import pandas as pd
# Save row after row just like algos-study

class Results:

    def __init__(self, resultDir):
        self.resultDir = resultDir
        try:
            self.r = pd.read_csv("{}/Results.csv".format(self.resultDir))
        except (FileNotFoundError, pd.errors.EmptyDataError):
            self.r = None

    def add_result(self, distance, s_acc, s_ber, t_acc, t_ber, sens_name, act_name, **params):
        """
        Save the given results.
        :param distance: the computed distance. Should be a dictionary containing the name of the metric and the result
        :param s_acc: the accuracy on the sensitive attribute
        :param s_ber: the ber on the sensitive attribute
        :param t_acc: the accuracy on the task
        :param t_ber: the ber on the task
        :param params: the parameters used for computing the sanitized sets.
        :return: an instance of the saved DataFrame.
        """
        # column attribute : sensitve and decisino
        # column clf : ....
        # column for each param
        # columns accuracies and columns bers

        to_list = lambda d, kn, vn: {kn: list(d.keys()), vn: list(d.values())}
        # def to_list(d, key_name, value_name):
        #     r = {
        #         key_name: [],
        #         value_name: [],
        #     }
        #     for k, v in d.items():
        #         r[key_name].append(k)
        #         r[value_name].append(v)
        #     return r
        def prepare(m1, m2, m1_attr, m2_attr, key, metric):
            s = to_list(m1, key, metric)
            s = pd.DataFrame.from_dict(s)
            s["Attribute"] = m1_attr

            t = to_list(m2, key, metric)
            t = pd.DataFrame.from_dict(t)
            t["Attribute"] = m2_attr

            r = pd.concat([s, t], axis=0)
            return r

        def prepare2(acc, tacc, ber, tber, sens_name, act_name, clfs="Classifiers"):
            r = {
                clfs: [],
                "Accuracies": [],
                "Ber": [],
                "Attribute": [],
            }
            # Respect the same order for every value
            for c in acc.keys():
                r[clfs].append(c)
                r["Accuracies"].append(acc[c])
                r["Ber"].append(ber[c])
                r["Attribute"].append(sens_name)

                r[clfs].append(c)
                r["Accuracies"].append(tacc[c])
                r["Ber"].append(tber[c])
                r["Attribute"].append(act_name)

            return pd.DataFrame.from_dict(r)

        # r = prepare(s_acc, t_acc, sens_name, act_name, "Classifiers", "Accuracies")
        # r_ = prepare(s_ber, t_ber, sens_name, act_name, "Classifiers", "Ber")
        # r = pd.concat([r, r_], axis=0)

        r = prepare2(s_acc, t_acc, s_ber, t_ber, sens_name, act_name, clfs='Classifiers')
        for k, v in distance.items():
            r["DistanceMetric"] = k
            r["DistanceValue"] = v

        for k, v in params.items():
            r[k] = v

        if self.r is None:
            self.r = r
        else:
            self.r = pd.concat([self.r, r], axis=0)

        self.r.to_csv("{}/Results.csv".format(self.resultDir), index=False)



class StopRestart:
    """Class for handling task kill and task restart (for metrics)"""

    def __init__(self, resultDir):
        """
        Init
        :param resultDir: Directory where results are saved
        """
        self.resultDir = resultDir
        self.can_check = True
        try:
            self.results = pd.read_csv("{}/Results.csv".format(self.resultDir))
        except (FileNotFoundError, pd.errors.EmptyDataError):
            self.can_check = False
            self.results = None

    def computed(self, **params):
        """
        Check if the given parameters exists in the CSV.
        :param params: parameters to check if computation is done.
        :return: True if the metric is computed for the given parameters
        """
        if self.can_check:
            # File exists. Check for params.
            r = self.results
            for p, v in params.items():
                try:
                    if np.isnan(v):
                        r = r[r[p].isna()]
                    else:
                        r = r[r[p] == v]
                except TypeError as e:
                    r = r[r[p] == v]
            return r.shape[0] != 0
        else:
            return False
