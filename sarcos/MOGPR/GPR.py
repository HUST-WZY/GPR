import gpytorch
import torch


from MOGPR.GPModel import MultitaskGPModel
from MOGPR.data_loader import loader

"""
Multitask GP Regression
"""


class GPR():

    def __init__(self, train_x, train_y, test_x, test_y, batch_size):

        self.train_x = train_x.cuda()
        self.train_y = train_y.cuda()

        self.test_x = test_x.cuda()
        self.test_y = test_y.cuda()

        self.batch_size = batch_size
        self.num_tasks = test_y.shape[1]

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.num_tasks).cuda()
        self.model = MultitaskGPModel(self.train_x[:batch_size], self.train_y[:batch_size], self.likelihood, self.num_tasks).cuda()


    def train(self, training_iterations, lr):

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(training_iterations):

            cur_x, cur_y = loader(self.train_x, self.train_y, self.batch_size)

            for j in range(len(cur_y)):

                batch_x = cur_x[j]
                batch_y = cur_y[j]

                optimizer.zero_grad()

                self.model.set_train_data(batch_x, batch_y)

                output = self.model(batch_x)
                loss = -mll(output, batch_y)
                loss.backward()

                print('Iter %d/%d - Batch Iter %d/%d - Loss: %.3f' % (
                    i + 1, training_iterations, j + 1, len(cur_y), loss.item()
                ))

                optimizer.step()




    def eval(self):

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():

            means_pred = self.likelihood(self.model(self.test_x))

            means_pred = means_pred.mean

        error = means_pred - self.test_y

        MAE = torch.zeros(self.num_tasks, 1)
        MSE = torch.zeros(self.num_tasks, 1)
        RMSE = torch.zeros(self.num_tasks, 1)

        for i in range(self.num_tasks):

            MAE[i] = torch.mean(torch.abs(error[:, i]))
            MSE[i] = torch.mean(torch.pow(error[:, i], 2))
            RMSE[i] = torch.sqrt(MSE[i])


        return MAE, MSE, RMSE, means_pred


def train_and_eval(train_x, train_y, test_x, test_y, batch_size, lr, epoch):

    gpr = GPR(train_x, train_y, test_x, test_y, batch_size)

    gpr.train(epoch, lr)

    MAE_g, MSE_g, RMSE_g, means_pred = gpr.eval()

    return MAE_g, MSE_g, RMSE_g, means_pred