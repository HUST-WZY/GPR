import gpytorch
import torch
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import SpectralMixtureKernel, ScaleKernel, GridInterpolationKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.models import ExactGP


class SMKISSGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SMKISSGPModel, self).__init__(train_x, train_y, likelihood)

        grid_size = gpytorch.utils.grid.choose_grid_size(train_x)

        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            GridInterpolationKernel(
                SpectralMixtureKernel(num_mixtures=4, ard_num_dims=train_x.shape[1]), grid_size=grid_size, num_dims=train_x.shape[1]
            )
        )

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)



class SMGP():

    def __init__(self, train_x, train_y, test_x, test_y):

        self.train_x = train_x.cuda()
        self.train_y = train_y.reshape(-1).cuda()
        self.test_x = test_x.cuda()
        self.test_y = test_y.reshape(-1).cuda()

        # self.batch_size = batch_size

        self.likelihood = GaussianLikelihood().cuda()
        self.model = SMKISSGPModel(self.train_x, self.train_y, self.likelihood).cuda()

    def train(self, lr, epoch, index_task, num_task):

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(epoch):

            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()

            print('Task %d/%d - Iter %d/%d - Loss: %.3f' % (
                index_task + 1, num_task, i + 1, epoch, loss.item()
            ))

            torch.cuda.empty_cache()

    def eval(self):

        self.model.eval()
        self.likelihood.eval()

        mean_pred = torch.tensor([0.])
        mean_true = torch.tensor([0.])

        with torch.no_grad(), gpytorch.settings.fast_pred_var():

            pred = self.likelihood(self.model(self.test_x))

            mean_pred = torch.cat([mean_pred, pred.mean.cpu()])
            mean_true = torch.cat([mean_true, self.test_y.cpu()])

        mean_pred = mean_pred[1:]
        mean_true = mean_true[1:]

        error = mean_pred - mean_true

        MAE = torch.mean(torch.abs(error))
        MSE = torch.mean(torch.pow(error, 2))
        RMSE = torch.sqrt(MSE)

        return MAE, MSE, RMSE, mean_pred


def train_and_eval(train_x, train_y, test_x, test_y, lr, epoch):
    # 输出维数
    num_task = test_y.shape[1]

    # 实例化深度 GPR 对象
    MAE_g = torch.zeros(1, num_task)
    MSE_g = torch.zeros(1, num_task)
    RMSE_g = torch.zeros(1, num_task)
    mean_pred = torch.zeros(test_x.shape[0], num_task)

    for i in range(num_task):

        deepGP = SMGP(train_x, train_y[:, i], test_x, test_y[:, i])
        deepGP.train(lr, epoch, i, num_task)
        MAE_g[0, i], MSE_g[0, i], RMSE_g[0, i], mean_pred[:, i] = deepGP.eval()
        torch.cuda.empty_cache()

    return MAE_g, MSE_g, RMSE_g, mean_pred