# DySan


This repository gathers the ressources (i.e., datasets and source codes) associated to the paper DySan: Dynamically sanitizing motion sensor data against sensitive inferences through adversarial networks. More precisely, we list the different datasets used in our evaluations and present a short guide to reproduce results for both DySan and the considered comparative baselines. If you have any question, do not hesitate to drop a message at dynamicsanitizer@gmail.com.


## Dependency
All dependencies are listed in the file lib.



## Datasets

We use reference datasets to evaluation DySan. We pre-processed the raw data of these datasets and we stored them in the directory called data. 


### MotionSense
The MotionSense dataset and a description of the data are publicly available: https://github.com/mmalekzadeh/motion-sense.
The training and testing set have been shaped thanks to the code available in the github of MotionSense.


### MobiAct
The MobiAct dataset and a description of the data are publicly available: https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/.



## Reproduce results

### Run DySan
To use DySan and to reproduce the experiments, you have to set the parameters in the file Parameters/Parameters.py in the DySan directory. Then you can run Sanitization.py to produce sanitized dataset. To produce results based on the sanitized dataset (i.e., compute the accuracy in term of activity recognition and sensitive attribute), you have to run Analysis.py.



Additionally, we also implemented a set of comparative baselines of the state-of-the-art to assess the performance of DySan. Here the list of these baselines:



### Run GEN 
GEN (Guardian-Estimator-Neutralizer) refers to the publication: **Malekzadeh, M., Clegg, R. G., Cavallaro, A., and Haddadi, H. Protecting sensory data against sensitive inferences. In W-P2DS’18, pp. 2:1–2:6, 2018**. 
While our implementation of this solution is based on DySan and uses a neural network with a slightly different architecture, we keep the same original behavior.
Similarly to DySan, to run GEN you have to configure the parameters in the file Parameters.py and then run Sanitization.py to produce the sanitized dataset. To evaluate the associated accuracy of the results, you have to run Analysis.py.




### Run OLYMPUS
OLYMPUS refers to the publication: **Raval, N., Machanavajjhala, A., and Pan, J. Olympus: Sensor privacy through utility aware obfuscation. In PETS, vol 1, 2019**. 
While our implementation of this solution is also based on DySan and uses a neural network with a slightly different architecture, we keep the same original behavior.
Similarly to DySan, to run GEN you have to configure the parameters in the file Parameters.py and then run Sanitization.py to produce the sanitized dataset. To evaluate the associated accuracy of the results, you have to run Analysis.py.



### Run MSDA
MSDA (Motion Sensor Data Anonymisation) refers to the publication: **Malekzadeh, M., Clegg, R. G., Cavallaro, A., and Haddadi, H., Mobile sensor data anonymization. In IoTDI 19, 2019**.
While our implementation of this solution is also based on DySan and uses a neural network with a slightly different architecture, we keep the same original behavior.
Similarly to DySan, to run GEN you have to configure the parameters in the file Parameters.py and then run Sanitization.py to produce the sanitized dataset. To evaluate the associated accuracy of the results, you have to run Analysis.py.


