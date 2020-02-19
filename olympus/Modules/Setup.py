import os

from Parameters import Parameters as P


def dump_exp_parameters(ExpDir, fairCoefs, sanM, discM, sanLoader, discLoader, sanLoss, discLoss, sanOptim, discOptim,
                        maxEpochs, SanIter, DiscIter, paramFile="ExperimentParams.txt"):
    # Just dumping every experiments parameters as strings
    with open("{}/{}".format(ExpDir, paramFile), 'a') as pf:
        pf.write("Fairness coeficients: \n")
        pf.write(str(fairCoefs))

        pf.write("\nModels: \n")
        pf.write("Sanitizer: \n")
        pf.write(str(sanM))
        pf.write("\nDiscriminator: \n")
        pf.write(str(discM))

        pf.write("\nLoaders: \n")
        pf.write("sanLoader: \n")
        pf.write(str(sanLoader))
        pf.write("\ndiscLoader: \n")
        pf.write(str(discLoader))

        pf.write("\nLosses: \n")
        pf.write("sanLosses: \n")
        pf.write(str(sanLoss))
        pf.write("\ndiscLosses: \n")
        pf.write(str(discLoss))

        pf.write("\nOptimizers: \n")
        pf.write("sanOptim: \n")
        pf.write(str(sanOptim))
        pf.write("\ndiscOptim: \n")
        pf.write(str(discOptim))

        pf.write("\nMax Epochs: {}\n".format(maxEpochs))
        pf.write("\nSan Iter: {}\n".format(SanIter))
        pf.write("\nDisc Iter: {}\n".format(DiscIter))


def mkSubFolders(expDir):
    try:
        for id in [P.ModelsDirId, P.FiguresDirId, P.ResultsDirId, P.ComputedStatisticsDirId,
                   P.GeneratedDataDirId]:
            os.makedirs("{}/{}/".format(expDir, id))
    except FileExistsError:
        pass

def prepare_experiment(baseDir, setBaseName, expNumber='auto', expName='Exp_', subExp='SubExp_',
                       endMarker='ExpCompleted', paramsDirId='Prm'):
    """
    Make new experiments directories
    :param setBaseName: The dataset base name
    :param expNumber: the experiment number. Set to 'auto' to auto compute the exp number. Gives an exp number to re-use
    sets of a given exp. If the folder does not exists, therefore create a new with the given number
    :param endMarker: if not None, check if the given exp number has completed before creating a new one. If None,
     create a new exp
    :return: The experiment base directory
    """

    exp = "{}/{}/{}".format(baseDir, setBaseName,expName)
    csvDir = None
    prmDir = None
    makeSplits = True

    if expNumber == 'auto':
        # Automatically add a new folder regardless of the other experiments status.
        counter = 1
        # if endMarker is None:
        # checker les exps et les sous-exps... trop complexe pour pas grand chose...
        while os.path.isdir("{}{}/".format(exp, counter)):
            counter += 1
        prmDir = "{}{}/{}".format(exp, counter, paramsDirId)
        os.makedirs(prmDir)
        exp = "{}{}/{}{}/".format(exp, counter, subExp, 1)
        mkSubFolders(exp)
    else:
        # Was designed to create a new subExp Folder such that all subexperiments share some parameters,
        # such as the data files. Check the status of the sub experiments and return the latest that has not ended yet,
        # or create a new sub exp if all have ended.
        exp = "{}{}/".format(exp, expNumber)
        prmDir = "{}/{}".format(exp, paramsDirId)
        if os.path.isdir(exp):
            # Folder exists, check for experiments. Csv exists
            counter = 1
            if endMarker is not None:
                while os.path.isdir("{}/{}{}/".format(exp, subExp, counter)) and \
                        os.path.isfile("{}/{}{}/{}".format(exp, subExp, counter, endMarker)):
                    counter += 1
                # If we end on a not completed exp, then will try to re-create already existing folders...
            else:
                while os.path.isdir("{}/{}{}/".format(exp, subExp, counter)):
                    counter += 1
            exp = "{}/{}{}/".format(exp, subExp, counter)
            makeSplits = False
            mkSubFolders(exp)
            # We ended on a not-completed exp, subFolders are already created. if FileExist is raised.

        else:
            # Folder does not exists, create it. Create csv and subExp.
            counter = 1
            os.makedirs(prmDir)
            exp = "{}/{}{}/".format(exp, subExp, counter)
            os.makedirs(exp)
            mkSubFolders(exp)

    return exp, prmDir, counter # Counter is either the exp number, or the subExp number if exp number is given

def makeCompletion(path, endMarker="ExpCompleted"):
    """
    Add a simple marker in the directory stating that the current experiment is completed
    :param path: the path where the experiments are located
    :param endMarker: the marker to add
    :return:
    """
    with open("{}/{}".format(path, endMarker), 'a') as ef:
        ef.write("Experiment Completed")
        print("Experiment Completed")

