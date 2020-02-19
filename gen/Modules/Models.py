import os
import torch
import torch.nn as nn
import torch.optim as O

class GeneralConv(nn.Module):
    """
    General methods for conv nets
    """

    def conv_len_out(self, lin, pad, ks, s):
        """
        Compute the length out for conv layers. See pytorch doc
        https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d

        Will update the pad if the length of output is a floating value.
        """
        def compute(lin, pad, ks, s):
            v = lin + 2 * pad - (ks -1) -1
            if (v % s != 0):
                v = v + (v%s)
                lout = (v / s) + 1
                pad = (v + 1 + (ks -1) - lin)
                if pad % 2 != 0:
                    pad += pad % 2
                return lout, pad, True
            else:
                lout = (v / s) + 1
                return lout, pad, False

        lout, pad, pad_modified = compute(lin, pad, ks, s)
        while pad_modified:
            lout, pad, pad_modified = compute(lin, pad, ks, s)

        return int(lout), int(pad)

    def pool_len_out(self, lin, pad, ks, s):
        """
        Compute the length out of the pooling layer. Will update the padding if the length is a floating value
        """
        def compute(lin, pad, ks, s):
            v = lin + 2 * pad - ks
            if (v % s != 0):
                v = v + (v%s)
                lout = (v / s) + 1
                pad = (v + ks - lin)
                if pad % 2 != 0:
                    pad += pad % 2
                return lout, pad, True
            else:
                lout = (v / s) + 1
                return lout, pad, False

        lout, pad, pad_modified = compute(lin, pad, ks, s)
        while pad_modified:
            lout, pad, pad_modified = compute(lin, pad, ks, s)

        return int(lout), int(pad)

    def pool_from_len_out_to_pad(self, lin, lout, ks, s):
        """
        Compute the necessary padding for obtaining the given len out, for the pooling layer
        """
        v = (lout - 1) * s + ks - lin
        assert v >= 0, "Cannot have a negative padding"
        if v % 2 != 0:
            # We cannot have a floating value for the padding, hence, we change to the nearest value above and we update
            # The length out.
            v += v % 2
            v = int(v / 2)
            lout, v = self.pool_len_out(lin, v, ks, s)
        else:
            v = v / 2
        return int(v), int(lout)

    def get_kernel_size(self, ks, seq_len, rev=1):
        # Compute the necessary kernel size for a suitable division
        if seq_len % 2 != 0:
            if ks % 2 != 0:
                ks += 1*rev
        else:
            if ks % 2 == 0:
                ks += 1*rev
        return int(ks)

class PredictorConv(GeneralConv):
    """
    Predictor of activities
    """
    def __init__(self, input_channels=6, seq_len=125, output_size=4, kernel_sizes=[5,5], strides=[1, 1],
                 conv_paddings=[0, 0], physNodes=0):

        super(PredictorConv, self).__init__()

        cOut = 100

        poolKernel = 2
        cpad = conv_paddings[0]
        kernel_sizes = kernel_sizes
        kernel_sizes[0] = self.get_kernel_size(kernel_sizes[0], seq_len)
        lout, cpad = self.conv_len_out(seq_len, cpad, kernel_sizes[0], strides[0])
        lout2 = int((lout // 2) + (lout % 2)) # Fix output len and get the necessary padding
        pool_pad, lout = self.pool_from_len_out_to_pad(lout, lout2, poolKernel, poolKernel)

        self.first = nn.Sequential(
            nn.Conv1d(input_channels, cOut, kernel_sizes[0], stride=strides[0], padding=cpad),
            nn.ReLU(),
            nn.AvgPool1d(poolKernel, stride=None, padding=pool_pad)
        )
        self.conv2_bn1 = nn.BatchNorm1d(cOut)

        cpad = conv_paddings[1]
        kernel_sizes[1] = self.get_kernel_size(kernel_sizes[1], lout)
        lout2, cpad = self.conv_len_out(lout, cpad, kernel_sizes[1], strides[1])
        pool_pad2, lout2 = self.pool_from_len_out_to_pad(lout2, int((lout2 // 2 ) + (lout2 % 2)), poolKernel, poolKernel)

        self.second = nn.Sequential(
            nn.Conv1d(cOut, cOut, kernel_sizes[1], stride=strides[1], padding=cpad),
            nn.ReLU(),
            nn.AvgPool1d(poolKernel, stride=None, padding=pool_pad2)

        )
	
        self.third = nn.Sequential(
            nn.Conv1d(cOut, 160, kernel_sizes[1], stride=strides[1], padding=cpad),
            nn.ReLU(),
            nn.AvgPool1d(poolKernel, stride=None, padding=pool_pad2)

        )

        self.fourth = nn.Sequential(
            nn.Conv1d(160, 160, kernel_sizes[1], stride=strides[1], padding=cpad),
            nn.ReLU(),
            nn.AvgPool1d(poolKernel, stride=None, padding=pool_pad2)

        )
        self.dropout = nn.Dropout(0.5)
        self.set_same_scale = nn.Tanh()
        # self.set_same_scale = lambda x: x

        cpad = conv_paddings[1]
        kernel_sizes[1] = self.get_kernel_size(kernel_sizes[1], lout)
        lout2, cpad = self.conv_len_out(lout, cpad, kernel_sizes[1], strides[1])
        pool_pad2, lout2 = self.pool_from_len_out_to_pad(lout2, int((lout2 // 2 ) + (lout2 % 2)), poolKernel, poolKernel)

        self.linears = nn.Sequential(
            nn.Linear(643, 64), #643
            nn.Linear(64, output_size),
            nn.Softmax(dim=1)
        )
        self.cat = lambda x, y: x
        if physNodes > 0:
            self.cat = lambda x, y: torch.cat((x, y), 1)

    def forward(self, input, phy=None):
        x = self.first(input)
        x = self.conv2_bn1(x) 
        x = self.second(x)
        x = self.third(x)
        x = self.fourth(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.cat(x, phy)
        x = self.set_same_scale(x)
        x = self.linears(x)
        return x


class DiscriminatorConv(GeneralConv):
    """
    Predictor of gender
    """
    def __init__(self, input_channels=6, seq_len=125, output_size=2, kernel_sizes=[5,5], strides=[1, 1], conv_paddings=[0, 0], physNodes=0):

        super(DiscriminatorConv, self).__init__()
        # For other models, do not do any pooling
        # Adapt for the given stride value.
        cOut = 256

        poolKernel = 2
        cpad = conv_paddings[0]
        kernel_sizes = kernel_sizes

        strides = [1,1]
        kernel_sizes[0] = self.get_kernel_size(kernel_sizes[0], seq_len)
        lout, cpad = self.conv_len_out(seq_len, cpad, kernel_sizes[0], strides[0])
        lout2 = int((lout // 2) + (lout % 2)) # Fix output len and get the necessary padding
        pool_pad, lout = self.pool_from_len_out_to_pad(lout, lout2, poolKernel, poolKernel)

        self.first = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_sizes[0], stride=strides[0], padding=cpad),
            nn.ReLU(),
            nn.AvgPool1d(poolKernel, stride=None, padding=pool_pad)
        )
        self.conv2_bn1 = nn.BatchNorm1d(64)

        cpad = conv_paddings[1]
        kernel_sizes[1] = self.get_kernel_size(kernel_sizes[1], lout)
        lout2, cpad = self.conv_len_out(lout, cpad, kernel_sizes[1], strides[1])
        pool_pad2, lout2 = self.pool_from_len_out_to_pad(lout2, int((lout2 // 2 ) + (lout2 % 2)), poolKernel, poolKernel)

        self.dropout = nn.Dropout(0.5)
        self.set_same_scale = nn.Tanh()
        # self.set_same_scale = lambda x: x
        self.linears1 = nn.Sequential(
            nn.Linear(60, 64),
            nn.Softmax(dim=1)
        )
        self.linears2 = nn.Sequential(
            nn.Linear(int(64 * 1 *  64)  + physNodes, output_size),
            nn.Softmax(dim=1)
        )
        self.cat = lambda x, y: x
        if physNodes > 0:
            self.cat = lambda x, y: torch.cat((x, y), 1)

    def forward(self, input, phy=None):
        x = self.first(input)
        x = self.conv2_bn1(x)
        x = self.linears1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.cat(x, phy)
        x = self.set_same_scale(x)
        x = self.linears2(x)
        return x


class ImcGenderConv(GeneralConv):

    def __init__(self, input_channels=6, seq_len=125, output_size=2, kernel_sizes=[5,5], strides=[1, 1], conv_paddings=[0, 0], physNodes=0):

        super(ImcGenderConv, self).__init__()
        cOut = 256
        poolKernel = 2
        cpad = conv_paddings[0]
        kernel_sizes = kernel_sizes
        strides = [1,1]
        kernel_sizes[0] = self.get_kernel_size(kernel_sizes[0], seq_len)
        lout, cpad = self.conv_len_out(seq_len, cpad, kernel_sizes[0], strides[0])
        lout2 = int((lout // 2) + (lout % 2)) # Fix output len and get the necessary padding
        pool_pad, lout = self.pool_from_len_out_to_pad(lout, lout2, poolKernel, poolKernel)

        self.first = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_sizes[0], stride=strides[0], padding=cpad),
            nn.ReLU(),
            nn.AvgPool1d(poolKernel, stride=None, padding=pool_pad)
        )
        self.conv2_bn1 = nn.BatchNorm1d(e) #A garder ???

        cpad = conv_paddings[1]
        kernel_sizes[1] = self.get_kernel_size(kernel_sizes[1], lout)
        lout2, cpad = self.conv_len_out(lout, cpad, kernel_sizes[1], strides[1])
        pool_pad2, lout2 = self.pool_from_len_out_to_pad(lout2, int((lout2 // 2 ) + (lout2 % 2)), poolKernel, poolKernel)

        self.dropout = nn.Dropout(0.5)
        self.set_same_scale = nn.Tanh()

        self.CNN = nn.Sequential(
            nn.Conv1d(64, 64, kernel_sizes[1], stride=strides[1], padding=cpad),
            nn.ReLU(),
            nn.AvgPool1d(poolKernel, stride=None, padding=pool_pad2)
        )
        self.linears = nn.Sequential(
            nn.Linear(int(lout2 * 1 *  64) + physNodes, 64),
            nn.Linear(64, output_size),
            nn.Softmax(dim=1)
        )
        self.cat = lambda x, y: x
        if physNodes > 0:
            self.cat = lambda x, y: torch.cat((x, y), 1)

    def forward(self, input, phy=None):
        x = self.first(input)
        x = self.conv2_bn1(x)
        x = self.CNN(x)
        x = x.view(x.size(0), -1)
        x = self.cat(x, phy)
        x = self.set_same_scale(x)
        x = self.linears(x)
        return x


def get_optimizer(model, lr=5e-4, wd=1e-10):
    """
    Get the models optimizers
    :param model: the model to use
    :param lr: The learning rate
    :param wd: weight rate decay
    """
    return O.Adam(params=model.parameters(), lr=lr, weight_decay=wd)


class SanitizerConv(GeneralConv):
    """
    Predictor of activities
    """
    def __init__(self, input_channels=6, seq_len=125, kernel_sizes=[5,5], strides=[1, 1],
                 conv_paddings=[0, 0], phyNodes=0, actNodes=0, noiseNodes=2):
        """

        :param input_channels:
        :param seq_len:
        :param output_size:


        :param phyNodes: Adding the nodes for the physiological data and the noise. If None, ignore them.

        If seq len is impair, then either use an impair value of kernel size with a pair value stride, which will
        necessitate an ajustment on the value of the conv padding,
        or
        simply use a pair value of kernel size. Otherwise, impossible to compute integer values of pooling and etc...
        THis is simply be cause we want to use the pool layer to divide the number of features by two.
        The same applies for pair values of seq_len
        """
        super(SanitizerConv, self).__init__()
        # For other models, do not do any pooling
        # Adapt for the given stride value.
        # Encoder
        cOut = 64
        kernel_sizes = kernel_sizes
        kernel_sizes[0] = self.get_kernel_size(kernel_sizes[0], seq_len, rev=1)
        lout, conv_paddings[0] = self.conv_len_out(seq_len, conv_paddings[0], kernel_sizes[0], strides[0])

        self.enc_first = nn.Sequential(
            nn.Conv1d(input_channels, cOut, kernel_sizes[0], stride=strides[0], padding=conv_paddings[0]),
            # nn.ReLU(),
        )
        kernel_sizes[1] = self.get_kernel_size(kernel_sizes[1], lout)
        lout2, conv_paddings[1] = self.conv_len_out(lout, conv_paddings[1], kernel_sizes[1], strides[1])

        self.enc_second = nn.Sequential(
            nn.Conv1d(cOut, 2 * cOut, kernel_sizes[1], stride=strides[1], padding=conv_paddings[1]),
            # nn.ReLU(),
        )
        # Conv output might be greater than 1. Physio data might be scale between 0(-1) and 1
        self.set_same_scale = nn.Tanh()

        self.enc_linears = nn.Sequential(
            nn.Linear(int(lout2 * 2 * cOut) + phyNodes + actNodes + noiseNodes, 128),
            nn.Linear(128, 64),
        )

        self.enc_dec_intermediate = nn.LeakyReLU()
        # Decoder
        self.dec_linear = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, int(lout2 * 2 * cOut) + phyNodes + actNodes),
        )
        # Kernel size and other have been computed before, only use the computed values. And the output
        # of the decoder should be of same shape as encoder input
        self.dec_second = nn.Sequential(
            nn.ConvTranspose1d(2 * cOut, cOut, kernel_sizes[1], stride=strides[1], padding=conv_paddings[1]),
            # nn.ReLU(),
        )
        self.dec_first = nn.Sequential(
            nn.ConvTranspose1d(cOut, input_channels, kernel_sizes[0], stride=strides[0], padding=conv_paddings[0]),
            # nn.ReLU(),
        )

        self.cat = lambda x, o: x
        self.uncat = lambda x: (x, None)
        nodes = phyNodes + actNodes
        if (nodes > 0) or (noiseNodes > 0):
            self.cat = lambda x, o: torch.cat((x, o), 1)
        if nodes > 0:
            # return sensor, act, phy
            self.uncat = lambda x: (x[:, :-(nodes)], x[:, -nodes:-phyNodes], x[:, -phyNodes:])
        self.act_activation = lambda x: x
        if actNodes > 0:
            self.act_activation = nn.Softmax(dim=1)


    def forward(self, input, others=None):
        """

        :param input: the sensor input for the forward pass
        :param others: the other data such as physiological, the activities and the noise, everything concatenated
        :return:
        """
        # Encode
        x = self.enc_first(input)
        x = self.enc_second(x)
        s = x.shape
        x = x.view(x.size(0), -1)
        # Will concatenate the phys and the noise
        x = self.cat(x, others)
        x = self.set_same_scale(x)
        x = self.enc_linears(x)

        # Intermediate
        x = self.enc_dec_intermediate(x)

        # Decode
        x = self.dec_linear(x)
        x = self.set_same_scale(x)
        # Will remove only the physio data. The noise is ignored.
        x, a, p = self.uncat(x)
        x = x.view(s)
        x = self.dec_second(x)
        x = self.dec_first(x)

        return x, self.act_activation(a), p



def save_classifier_states(NN, version, directory, otherParamFn, ext="S"):
    """
    Save given classifier parameter
    :param NN: Classifier to save
    :param version: classifier parameter version (epoch number)
    :param directory: destination directory, where to save parameters.
    :param otherParamFn: Function to add other parameters on the name of the model
    :return: False if there is an error. True otherwise
    """
    if otherParamFn is None:
        otherParamFn = lambda x: x
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(NN.state_dict(), "{d}/Epoch_{v}.{e}".format(d=directory, v=otherParamFn(version), e=ext))

def save_classifier_states2(NN, name):
    """
    Save given classifier parameter
    :param NN: Classifier to save
    :param version: classifier parameter version (epoch number)
    :param directory: destination directory, where to save parameters.
    :param otherParamFn: Function to add other parameters on the name of the model
    :return: False if there is an error. True otherwise
    """

    torch.save(NN.state_dict(), name + "_Model")

def get_latest_consistent_epoch(model_dir, san_ext="S", disc_ext="D", pred_ext="P", otherParamFn=None):
    """
    Return the latest epoch where both the sanitizer and all predictors are available
    """
    epoch = 0
    stop = False
    c_format = "{d}/Epoch_{v}.{e}"
    while not stop:
        epoch += 1
        stop = not os.path.isfile(c_format.format(d=model_dir, v=otherParamFn(epoch), e=san_ext))
        stop = not os.path.isfile(c_format.format(d=model_dir, v=otherParamFn(epoch), e=disc_ext)) or stop
        stop = not os.path.isfile(c_format.format(d=model_dir, v=otherParamFn(epoch), e=pred_ext)) or stop
    return epoch - 1


def get_latest_states(model_dir, san, disc, pred, san_ext="S", disc_ext="D", pred_ext="P", otherParamFn=None):
    epoch = get_latest_consistent_epoch(model_dir=model_dir, san_ext=san_ext, disc_ext=disc_ext, pred_ext=pred_ext,
                                        otherParamFn=otherParamFn)
    if epoch > 0:
        load_classifier_state(san, epoch, model_dir, san_ext, otherParamFn)
        load_classifier_state(disc, epoch, model_dir, disc_ext, otherParamFn)
        load_classifier_state(pred, epoch, model_dir, pred_ext, otherParamFn)
    return epoch +1

def load_classifier_state(NN, epoch, model_dir, ext, otherParamFn):
    """
    Load classifier parameter
    :param NN: Classifier where to load parameters.
    :param epoch: epoch to load
    :param model_dir: directory where to load epochs. First directory
    """
    NN.load_state_dict(torch.load("{d}/Epoch_{v}.{e}".format(d=model_dir, v=otherParamFn(epoch), e=ext)))


def load_classifier_state2(NN, name):
    """
    Load classifier parameter
    :param NN: Classifier where to load parameters.
    :param epoch: epoch to load
    :param model_dir: directory where to load epochs. First directory
    """
    NN.load_state_dict(torch.load(name + "_Model"))


