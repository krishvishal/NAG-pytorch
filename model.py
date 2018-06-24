import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules import Module


class VirtualBatchNorm2d(Module):
    """
    Module for Virtual Batch Normalization.
    Implementation borrowed and modified from Rafael_Valle's code + help of SimonW from this discussion thread:
    https://discuss.pytorch.org/t/parameter-grad-of-conv-weight-is-none-after-virtual-batch-normalization/9036
    """
    def __init__(self, num_features: int, eps: float=1e-5):
        super().__init__()
        # batch statistics
        self.num_features = num_features
        self.eps = eps  # epsilon
        self.ref_mean = self.register_parameter('ref_mean', None)
        self.ref_mean_sq = self.register_parameter('ref_mean_sq', None)

        # define gamma and beta parameters
        gamma = torch.normal(means=torch.ones(1, num_features, 1), std=0.02)
        self.gamma = Parameter(gamma.float().cuda(async=True))
        self.beta = Parameter(torch.cuda.FloatTensor(1, num_features, 1).fill_(0))

    def get_stats(self, x):
        """
        Calculates mean and mean square for given batch x.
        Args:
            x: tensor containing batch of activations
        Returns:
            mean: mean tensor over features
            mean_sq: squared mean tensor over features
        """
        mean = x.mean(2, keepdim=True).mean(0, keepdim=True)
        mean_sq = (x ** 2).mean(2, keepdim=True).mean(0, keepdim=True)
        return mean, mean_sq

    def forward(self, x, ref_mean: None, ref_mean_sq: None):
        """
        Forward pass of virtual batch normalization.
        Virtual batch normalization require two forward passes
        for reference batch and train batch, respectively.
        The input parameter is_reference should indicate whether it is a forward pass
        for reference batch or not.
        Args:
            x: input tensor
            is_reference(bool): True if forwarding for reference batch
        Result:
            x: normalized batch tensor
        """
        mean, mean_sq = self.get_stats(x)
        if ref_mean is None or ref_mean_sq is None:
            # reference mode - works just like batch norm
            mean = mean.clone().detach()
            mean_sq = mean_sq.clone().detach()
            out = self._normalize(x, mean, mean_sq)
        else:
            # calculate new mean and mean_sq
            batch_size = x.size(0)
            new_coeff = 1. / (batch_size + 1.)
            old_coeff = 1. - new_coeff
            mean = new_coeff * mean + old_coeff * ref_mean
            mean_sq = new_coeff * mean_sq + old_coeff * ref_mean_sq
            out = self._normalize(x, mean, mean_sq)
        return out, mean, mean_sq

    def _normalize(self, x, mean, mean_sq):
        """
        Normalize tensor x given the statistics.
        Args:
            x: input tensor
            mean: mean over features. it has size [1:num_features:]
            mean_sq: squared means over features.
        Result:
            x: normalized batch tensor
        """
        assert mean_sq is not None
        assert mean is not None
        assert len(x.size()) == 4  # specific for 1d VBN
        if mean.size(1) != self.num_features:
            raise Exception(
                    'Mean size not equal to number of featuers : given {}, expected {}'
                    .format(mean.size(1), self.num_features))
        if mean_sq.size(1) != self.num_features:
            raise Exception(
                    'Squared mean tensor size not equal to number of features : given {}, expected {}'
                    .format(mean_sq.size(1), self.num_features))

        std = torch.sqrt(self.eps + mean_sq - mean**2)
        x = x - mean
        x = x / std
        x = x * self.gamma
        x = x + self.beta
        return x

    def __repr__(self):
        return ('{name}(num_features={num_features}, eps={eps}'
                .format(name=self.__class__.__name__, **self.__dict__))


def make_z(shape, minval, maxval):

	z = minval + torch.rand(shape) * (maxval - 1)
	return z

class Generator(nn.Module):
	def __init__(self, batch_size=32, image_size=[3,32,32], y_dim=None,
				 z_dim=100, gf_dim=64, df_dim=64,
				 out_init_b=0., out_stddev=.15):
		super().__init__()

		self.batch_size = batch_size
		self.z_dim = z_dim
		self.gf_dim  = gf_dim
		self.df_dim = df_dim
		self.out_init_b = out_init_b
		self.out_stddev = out_stddev
		self.y_dim = y_dim
		self.image_shape = image_shape

	# def generate(self, is_ref=False):

		# def make_z(shape, minval, maxval):

		# 	z = (minval - maxval) * torch.rand(shape) + maxval
		# 	return z 

		self.linear1 = nn.Linear(100, self.gf_dim*7*4*4)
		self.vbn1 = VirtualBatchNorm2d(self.gf_dim*7)
		
		self.deconv1 = nn.ConvTranspose2d(512, self.gf_dim*4, kernel_size=5, stride=2, padding=2)


		self.deconv2 = nn.ConvTranspose2d(320, self.gf_dim*2, kernel_size=5, stride=2, padding=2)
		self.vbn2 = VirtualBatchNorm2d(self.gf_dim*4)

		self.deconv3 = nn.ConvTranspose2d(160, self.gf_dim*1, kernel_size=5, stride=2, padding=2)
		self.vbn3 = VirtualBatchNorm2d(self.gf_dim*2)

		self.deconv4 = nn.ConvTranspose2d(80, self.gf_dim*1, kernel_size=5, stride=2, padding=2)
		self.vbn4 = nn.VirtualBatchNorm2d(self.gf_dim*1)

		self.deconv5 = nn.ConvTranspose2d(72, self.gf_dim*1, kernel_size=5 ,stride=2, padding=2)
		self.vbn5 = nn.VirtualBatchNorm2d(self.gf_dim*1)

		self.deconv6 = nn.ConvTranspose2d(72, self.gf_dim*1, kernel_size=5, stride=2, padding=2)
		self.vbn6 = nn.VirtualBatchNorm2d(self.gf_dim*1)

		self.deconv7 = nn.ConvTranspose2d(72, 3, kernel_size=5, stride=1, padding=2)
		self.vbn7 = nn.VirtualBatchNorm2d(self.gf_dim*1)
		torch.nn.init.normal(self.deconv7.weight, std = self.out_stddev)
		torch.nn.init.constant(self.deconv7.bias, 0.1)

		self.tanh = nn.Tanh()

	def forward(self, ref_x, x):

##########################################################################################################
#  Forward pass using reference batch                                                                    #
##########################################################################################################

		z_ = self.linear(ref_x)
		h0 = z_.view(-1, 64*7, 4, 4)
		h0 = F.relu(h0)
		h0, mean0, meansq0 = self.vbn1(h0, None, None)
		h0z = make_z((self.batch_size, self.gf_dim, 4, 4), minval=-1., maxval=1.)
		h0 = torch.cat((h0, h0z), 1)

		h1 = self.deconv1(h0)
		h1, mean1, meansq1 = self.vbn2(h1, None, None)
		h1 = F.relu(h1)
		h1z = make_z((self.batch_size, self.gf_dim, 7, 7), minval=-1., maxval=1.)
		h1 = torch.cat((h1, h1z), 1)

		h2 = self.deconv2(h1, output_size=(10, 128, 14, 14))
		h2, mean2, meansq2 = self.vbn3(h2, None, None)
		h2 = F.relu(h2)
		half = self.gf_dim // 2
		if half == 0:
			half = 1
		h2z = make_z((self.batch_size, half, 14, 14), minval=-1., maxval=1.)
		h2 = torch.cat((h2, h2z), 1)

		h3 = self.deconv3(h2, output_size=(10, 64, 28, 28))
		h3, mean3, meansq3 = self.vbn4(h3, None, None)
		h3 = F.relu(h3)
		quarter = self.gf_dim // 4
		if quarter == 0:
			quarter = 1
		h3z = make_z((self.batch_size, quarter, 28, 28), minval=-1., maxval=1.)
		h3 = torch.cat((h3, h3z), 1)

		h4 = self.deconv4(h3, output_size=(10, 64, 56, 56))
		h4, mean4, meansq4 = self.vbn5(h4, None, None)
		h4 = F.relu(h4)
		eighth = self.gf_dim // 8
		if eighth == 0:
			eighth = 1
		h4z = make_z((self.batch_size, eighth, 56, 56), minval=-1., maxval=1.)
		h4 = torch.cat((h4, h4z), 1)

		h5 = self.deconv5(h, output_size=(10, 64, 112, 112))
		h5, mean5, meansq5 = self.vbn6(h5, None, None)
		h5 = F.relu(h5)
		sixteenth = self.gf_dim // 16
		if sixteenth == 0:
			sixteenth = 1
		h5z = make_z((self.batch_size, eighth, 112, 112), minval=-1., maxval=1.)
		h5 = torch.cat((h5, h5z), 1)

		h6 = self.deconv6(h5, output_size=(10, 64, 224, 224))
		h6, mean6, meansq6 = self.vbn7(h6, None, None)
		h6 = F.relu(h6)
		sixteenth = self.gf_dim // 16
		if sixteenth == 0:
			sixteenth = 1
		h6z = make_z((self.batch_size, eighth, 224, 224), minval=-1., maxval=1.)
		h6 = torch.cat((h6, h6z), 1)

		h7 = self.deconv7(h6, output_size=(10, 3, 224, 224))
		out = 10*self.tanh(h7)



##########################################################################################################
#  The real Forward pass using the parameters from reference forward pass                                #
##########################################################################################################
		
		z_ = self.linear(x)
		h0 = z_.view(-1, 64*7, 4, 4)
		h0, _, _ = self.vbn1(h0, mean0, meansq0)
		h0 = F.relu(h0)
		h0z = make_z((self.batch_size, self.gf_dim, 4, 4), minval=-1., maxval=1.)
		h0 = torch.cat((h0, h0z), 1)

		h1 = self.deconv1(h0)
		h1, _, _ = self.vbn2(h1, mean1, meansq1)
		h1 = F.relu(h1)
		h1z = make_z((self.batch_size, self.gf_dim, 7, 7), minval=-1., maxval=1.)
		h1 = torch.cat((h1, h1z), 1)

		h2 = self.deconv2(h1, output_size=(10, 128, 14, 14))
		h2, _, _ = self.vbn3(h2, mean2, meansq2)
		h2 = F.relu(h2)
		half = self.gf_dim // 2
		if half == 0:
			half = 1
		h2z = make_z((self.batch_size, half, 14, 14), minval=-1., maxval=1.)
		h2 = torch.cat((h2, h2z), 1)

		h3 = self.deconv3(h2, output_size=(10, 64, 28, 28))
		h3, _, _ = self.vbn4(h3, mean3, meansq3)
		h3 = F.relu(h3)
		quarter = self.gf_dim // 4
		if quarter == 0:
			quarter = 1
		h3z = make_z((self.batch_size, quarter, 28, 28), minval=-1., maxval=1.)
		h3 = torch.cat((h3, h3z), 1)

		h4 = self.deconv4(h3, output_size=(10, 64, 56, 56))
		h4, _, _ = self.vbn5(h4, mean4, meansq4)
		h4 = F.relu(h4)
		eighth = self.gf_dim // 8
		if eighth == 0:
			eighth = 1
		h4z = make_z((self.batch_size, eighth, 56, 56), minval=-1., maxval=1.)
		h4 = torch.cat((h4, h4z), 1)

		h5 = self.deconv5(h, output_size=(10, 64, 112, 112))
		h5, _, _ = self.vbn6(h5, mean5, meansq5)
		h5 = F.relu(h5)
		sixteenth = self.gf_dim // 16
		if sixteenth == 0:
			sixteenth = 1
		h5z = make_z((self.batch_size, eighth, 112, 112), minval=-1., maxval=1.)
		h5 = torch.cat((h5, h5z), 1)

		h6 = self.deconv6(h5, output_size=(10, 64, 224, 224))
		h6, _, _ = self.vbn7(h6, mean6, meansq6)
		h6 = F.relu(h6)
		sixteenth = self.gf_dim // 16
		if sixteenth == 0:
			sixteenth = 1
		h6z = make_z((self.batch_size, eighth, 224, 224), minval=-1., maxval=1.)
		h6 = torch.cat((h6, h6z), 1)

		h7 = self.deconv7(h6, output_size=(10, 3, 224, 224))
		out = 10*self.tanh(h7)

		return out





