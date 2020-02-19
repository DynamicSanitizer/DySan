import torch
import numpy as np
import torch.nn as nn


# Redefine function such as nllloss if you wish to use them as loss. See the example below
# Note that the shape of the given input is [mxn] and the target is [m] (just as for NLLLoss).
# See Accuracy for example to see how to handle the specific format.

class NLLLoss(nn.NLLLoss):
    """
    Redefinition of NLLLoss, just to take into account specificities in the forward pass
    """

    def __init__(self, *args, **kwargs):
        """
        Take exactly the same arguments as the nn.NLLLoss (see:
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html)
        """
        super(NLLLoss, self).__init__(*args, **kwargs)

    def forward(self, input, target, *args, **kwargs):
        """
        Exactly as the nll loss, we simply ignore the other attributes
        """
        return super().forward(input, target)


class AccuracyLoss(nn.Module):
    """
    Compute the accuracy loss, in a compatible way as nn.Module (Mostly for CUDA)
    """

    def __init__(self, device="cpu"):
        super(AccuracyLoss, self).__init__()

    def forward(self, input, target, *args, **kwargs):
        """
        Compute the accuracy
        """
        # Selecting the outputs that should be maximized (equal to 1)
        out = input[range(len(target)), target]
        # computing accuracy loss (distance to 1
        return 1 - out.mean()


class CircularAccuracyLoss(nn.Module):
    """
    Compute a circular accuracy loss: the target is step away from the real target value. If a value is above a maximum,
    then we map it back to the minimum value.
    """

    def __init__(self, max_=3, step=2, device="cpu"):
        """
        Initialization
        :param max_: the maximum value of all possible targets
        :param step: the difference between the real value and the wished target
        :param device: the device on which to compute the results.
        """
        super(CircularAccuracyLoss, self).__init__()
        self.max_ = max_ + 1
        self.step = step
        self.device = device
        self.fn = AccuracyLoss(device)

    def forward(self, input, target, *args, **kwargs):
        """
        Compute the loss
        :param input: the predicted values
        :param target: the real value without any modifications, straight out of the dataset
        """
        t = (target + self.step) % self.max_
        t.to(self.device)
        return self.fn(input, t, *args, **kwargs)


class BalancedErrorRateLoss(nn.Module):
    """
    Compute the balanced error rate loss.
    """

    def __init__(self, targetBer=1 / 2, device="cpu"):
        """
        :param targetBer: the value of the BER to be closed to
        """
        super(BalancedErrorRateLoss, self).__init__()
        self.targetBer = targetBer
        self.device = device

    def get_true_value(self, computed_ber):
        """
        Return the true value of the computed BER, not the distance from target. BER is between [0,1/2]. Impossible to
        go beyond that interval unless wrongly implemented.
        :param computed_ber: the distance from the target BER. Must be of type numpy array.
        """
        if isinstance(computed_ber, list):
            computed_ber = np.array(computed_ber)
        b = -computed_ber + self.targetBer
        return np.abs(b)

    def forward(self, input, target, sens, *args, **kwargs):
        """
        Comput the balanced error rate
        :param input: the input
        :param target: the target values
        :param sens: the sensitive attribute
        :param args: the argument to ignore
        :param kwargs: the keyword arguments to ignore
        :return: the computed loss
        """
        # Selecting the right predictions,
        out = input[range(len(target)), target]
        # Computing errors
        out = torch.abs(1 - out)
        # Reshaping data
        sens.to(self.device)
        sens = sens.view(-1)
        out = out.view(-1, 1)
        l = len(out)
        # Summing by mask values
        # From: https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335
        m = torch.zeros(sens.max() + 1, l).to(self.device)
        m[sens, torch.arange(l)] = 1
        m = nn.functional.normalize(m, p=1, dim=1)
        # Computing the mean of each group\
        out = torch.mm(m, out)
        out = out.mean()
        # k = torch.abs(input.argmax(1) - target).type(torch.FloatTensor).to(self.device)
        # k = (k[sens==1].mean() + k[sens==0].mean())/2
        return torch.abs(self.targetBer - out)


class GeneralSanitizerLoss(nn.Module):
    """
    Loss function for the Sanitizer
    """

    def __init__(self, alpha_=0.25, lambda_=0.25, recOn=True, optim_type="mean", device="cpu"):
        """
        Init.
        :param optim_type: The optimisation type to consider, either a sum or a vector.
        Values are: "mean", "vector", "model_vector".
        Mean will compute the mean of all losses
        Vector: will set the function as a vector of attributes, with the predictor and the discriminator losses
         as separate components
        Model_vector: Will compute the loss as three vector, each one for each corresponding model

        :param device: the device the compute on
        """
        super(GeneralSanitizerLoss, self).__init__()
        self.pred_loss = None
        self.disc_loss = None
        self.compute = False

        if "vector" == str(optim_type).lower():
            self.ae_loss = lambda x, y: nn.L1Loss(reduction="none")(x, y).sum(0).sum(1) / (x.size(0) * x.size(2))
            self.combine = lambda task, sens, sensor: \
                torch.cat((lambda_ * task.view(1, -1), alpha_ * sens.view(1, -1),
                           (1 - (alpha_ + lambda_)) * sensor.view(1, -1)), 1)
        elif "model_vector" == str(optim_type).lower():
            self.ae_loss = nn.L1Loss()
            self.combine = lambda task, sens, sensor: \
                torch.cat((lambda_ * task.view(1, -1), alpha_ * sens.view(1, -1),
                           (1 - (alpha_ + lambda_)) * sensor.mean().view(1, -1)), 1)
        else:
            self.ae_loss = nn.L1Loss()
            self.combine = lambda task, sens, sensor: (lambda_ * task + alpha_ * sens + (1 - (alpha_ + lambda_)) *
                                                       sensor.mean())

        if not recOn:
            # We do not care about the reconstruction loss. return a random 0 for the reconstruction.
            self.ae_loss = lambda *args, **kwargs: torch.zeros(1, requires_grad=True).to(device)

    def forward(self, sensor_s, other_s, act_p, sens_p, sensor, act, sens, other, *args, **kwargs):
        """
        compute the loss
        :param sensor_s: the sanitized sensor values
        :param other_s: other sanitized attributes such as the physiological and the sanitized activities
        :param act_p: the predicted activities
        :param sens_p: the predicted sensitive attribute
        :param sensor: the original sensor values
        :param act: the target activities
        :param sens: the target sensitive
        :param other: the target other attributes
        :return: the computed loss
        """
        if self.compute:
            # sensitive
            sens_loss = self.disc_loss(input=sens_p, target=sens, sens=sens)
            # activities
            act_loss = self.pred_loss(input=act_p, target=act, sens=sens)
            # Sensor losses
            sensor_loss = self.ae_loss(sensor_s, sensor)
            # Physio loss, Reconstructed activities
            physio_loss = self.ae_loss(other_s.view(*other.size(), 1), other.view(*other.size(), 1))
            san_loss = torch.cat((sensor_loss.view(1, -1), physio_loss.view(1, -1)), 1)

            return self.combine(task=act_loss, sens=sens_loss, sensor=san_loss), act_loss, sens_loss

        raise NotImplementedError(
            "Can not use {} directly. Please create a proper class".format(self.__class__.__name__))


class SanitizerBerLoss(GeneralSanitizerLoss):
    """
    Loss function for the Sanitizer
    """

    def __init__(self, alpha_=0.25, lambda_=0.25, recOn=True, optim_type="mean", device="cpu"):
        """
        Init.
        :param optim_type: The optimisation type to consider, either a sum or a vector.
        Values are: "mean", "vector", "model_vector".
        Mean will compute the mean of all losses
        Vector: will set the function as a vector of attributes, with the predictor and the discriminator losses
         as separate components
        Model_vector: Will compute the loss as three vector, each one for each corresponding model

        :param device: the device the compute on
        """
        super(SanitizerBerLoss, self).__init__(alpha_=alpha_, lambda_=lambda_, recOn=recOn, optim_type=optim_type,
                                               device=device)
        # self.pred_loss = AccuracyLoss(device=device)
        self.pred_loss = BalancedErrorRateLoss(targetBer=0, device=device)
        self.disc_loss = BalancedErrorRateLoss(targetBer=1 / 2, device=device)
        self.compute = True


class SanitizerCircularLoss(GeneralSanitizerLoss):
    """
    Loss function for the Sanitizer
    """

    def __init__(self, alpha_=0.25, lambda_=0.25, max_=3, step=2, recOn=True, optim_type="mean", device="cpu"):
        """
        Init.
        :param optim_type: The optimisation type to consider, either a sum or a vector.
        Values are: "mean", "vector", "model_vector".
        Mean will compute the mean of all losses
        Vector: will set the function as a vector of attributes, with the predictor and the discriminator losses
         as separate components
        Model_vector: Will compute the loss as three vector, each one for each corresponding model

        :param device: the device the compute on
        :param max_: the maximum value of all possible targets
        :param step: the difference between the real value and the wished target
        """
        super(SanitizerCircularLoss, self).__init__(alpha_=alpha_, lambda_=lambda_, recOn=recOn, optim_type=optim_type,
                                                    device=device)
        # self.pred_loss = AccuracyLoss(device=device)
        self.pred_loss = BalancedErrorRateLoss(targetBer=0, device=device)
        self.disc_loss = CircularAccuracyLoss(max_=max_, step=step, device=device)
        self.compute = True
