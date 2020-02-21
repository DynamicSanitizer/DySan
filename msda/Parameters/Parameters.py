#### DIRECTORIES ####
ExperimentBaseDir = "./Experiments/"
# Experience Number. Set to "auto" to auto create exp dir. Gives a specific number to continue an experiment
ExpNumber = "1"
# SubFolders
ModelsDirId = "Models"
FiguresDirId = "Figures"
ResultsDirId = "Results"
ComputedStatisticsDirId = "Stats"
GeneratedDataDirId = "Generated"
ParamsDirId = "Prm"

#### Datasets
SetName = "msda"
TrainPath =  "../data/motionsense_trainTrial.csv"
TestPath = "../data/motionsense_testTrial.csv"

# Column excluded from the preprocessing process, they will be appended at the of the preprocessed dataset as it
PreprocessingExcluded = ["trial", "act", "id", "gender"]
# Include overwrite excluded. Set to None or Empty list to avoid overwriting
PreprocessingIncluded = ["height", "weight", "age"]
# PreprocessingIncluded = None
# Preprocessing lower bound of values
Scale = -1
# Window overlap in seconds
Window_overlap = 1.25

#### Coefficients
Alpha = 0.6
Lambda = 0.3
# Set to True if we care about the distance. False if not. Make sure not to use any of the sanitized output in
# subsequent computation. (e.g: see the auto check at the end)
RecOn = True

#### Loss functions (None by default)
SanLoss = None
PredLoss = None
DiscLoss = None

#### Max iterations of discriminator and predictor
KPred = 50
KDisc = 50

### Reset the predictor and discriminator states each epoch ?
TrainingResetModelsStates = False

### Epochs
Epoch = 300
# Plot the loss every <PlotRate> epochs
PlotRate = 1

### Batch size
# Use a single batch size for all models
BatchSize = 256

### Optimization type
# Values are: "mean", "vector", "model_vector".
# Mean will compute the mean of all losses
# Vector: will set the function as a vector of attributes, with the predictor and the discriminator losses
# as separate components
# Model_vector: Will compute the loss as three vector, each one for each corresponding model
OptimType = "vector"

### Noise
NoiseNodes = 2
NoiseGenerator = {"p": "torch", "f":"rand"}

### Use Physiological data in input ?
PhysInput = True
# During data generation (not the sanitization), which type of data (original or sanitized) should we consider ?
SanitizePhysio = False
SanitizeActivities = False
# If the discriminator cannot predict the sens attr even with the activity, then the activity has no info about sens.
DecorrelateActAndSens = True

#### Which activity to use as input of the discriminator ? The sanitized or the original activity ?
# Using the original means that from the sensor info and even complementary info such as the unperturbed act,
# we are unable to predict the sensitive attribute (this gives the lower bound). Using the sanitized act, we have
# the upper bound
# format is parameters : function return
ActivitySelection = lambda original, sanitized: original
PhysiologSelection = lambda original, sanitized: sanitized

##### Metric
# Distance metric used between the original and sanitized sets
DistanceMetric = "euclidean"








## Path auto generated. Values are set automatically. Useless to modify here.
ExpPath = None
PrmPath = None
ModelsDir = lambda : "{}/{}/".format(ExpPath, ModelsDirId) # Setting as functions since Exp and Prm are dynamically
# modified
FiguresDir = lambda : "{}/{}/".format(ExpPath, FiguresDirId)
ResultsDir = lambda : "{}/{}/".format(ExpPath, ResultsDirId)
CompStatDir = lambda : "{}/{}/".format(ExpPath, ComputedStatisticsDirId)
GenDataDir = lambda : "{}/{}/".format(ExpPath, GeneratedDataDirId)

### Function to combine parameters into a single string.
ParamFunction = lambda epoch: "A={}-L={}-O={}-KP={}-KD={}-NN={}-Rec={}-E={}"\
    .format(Alpha, Lambda, OptimType, KPred, KDisc, NoiseNodes,"On" if RecOn else "Off", epoch)



##### Auto check. Check the compatibility of some parameters. Set the value in the correct way
if not RecOn:
    # Do not use sanitized output. Use the original ones.
    # RecOn = False can be achieved by setting Alpha + Lambda == 1, however, we will still compute gradient and multiply
    # them to the 0 coefficient. It might be computationally inefficient compared to setting RecOn = False.
    ActivitySelection = lambda original, sanitized: original
    PhysiologSelection = lambda original, sanitized: original

assert (Alpha + Lambda) <= 1, "The sum of Alpha + Lambda must be less than 1"
