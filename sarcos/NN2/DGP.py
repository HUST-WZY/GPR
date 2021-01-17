import torch
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel, SpectralMixtureKernel
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.mlls import VariationalELBO, DeepApproximateMLL
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.settings import num_likelihood_samples
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch.nn import Linear
from torch.utils.data import TensorDataset, DataLoader
import tqdm


class DGP():
    """
    深度GPR最终的子类，多输入多输出
    在实际使用直接示例化为对象调用
    在实例化时，输入训练及测试数据集即可
    注意数据集都应该包含两个维度

    但是目前只能在 cpu 上跑，使用 gpu 会报错
    """

    def __init__(self, train_x, train_y, test_x, test_y):

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

        # likelihood 无需手动初始化，为 MyDeepGP 对象的属性
        self.model = MyDeepGP(train_x.shape[1], train_y.shape[1])


    def train(self, lr, train_iter, batch_size):
        """
        训练方法
        :param lr: 学习率
        :param train_iter: 迭代次数，即训练的 Epoch 的个数
        :param batch_size: 批的大小
        :return:
        """

        self.model.train()

        # 由于是在 cpu 上跑，DataLoader 可以用
        train_dataset = TensorDataset(self.train_x, self.train_y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.SGD([
            {'params': self.model.parameters()},
        ], lr=lr)

        # 最后一个值为训练数据的个数
        mll = DeepApproximateMLL(VariationalELBO(self.model.likelihood, self.model, self.train_x.shape[-2]))

        epochs_iter = tqdm.tqdm(range(train_iter), desc="Epoch")

        for i in epochs_iter:

            minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)

            for batch_x, batch_y in minibatch_iter:

                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = -mll(output, batch_y)
                loss.backward()
                optimizer.step()

                minibatch_iter.set_postfix(loss=loss.item())


    def eval(self, batch_size, num_samples):
        """
        测试方法
        :param batch_size: 批的大小
        :param num_samples: 在预测时，采样点的个数
        :return:
        """

        self.model.eval()

        MAE, MSE, RMSE, mean_pred = self.model.predict(self.test_x, self.test_y, batch_size, num_samples)

        return MAE, MSE, RMSE, mean_pred


class MyGPLayer(DeepGPLayer):
    """
    GP 神经网络层 类
    该类的实例化对象即为 GP 神经网络中的一层
    在示例化时需要指定该层 输入的维数、输出的维数、均值函数的类型
    """

    def __init__(self, input_dims, output_dims=None, num_inducing=128, mean_type='constant'):

        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        # 变分策略
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(MyGPLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)

        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

        self.linear_layer = Linear(input_dims, 1)

    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        为了可以像 ResNET，那样，加了这个
        :param x:
        :param other_inputs:
        :param kwargs:
        :return:
        """
        if len(other_inputs):
            if isinstance(x, MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(self.num_samples, *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


class MyDeepGP(DeepGP):
    """
    深度GP神经网络类
    通过实例化 MyGPLayer 的对象，一层层堆积成神经网络
    本类的对象在示例化时需要指定多输入多输出的维数
    """

    def __init__(self, input_dims, output_dims):

        # 隐藏层1
        hidden_layer1 = MyGPLayer(
            input_dims=input_dims,
            output_dims=10,
            mean_type='linear'
        )

        # 隐藏层2
        hidden_layer2 = MyGPLayer(
            input_dims=hidden_layer1.output_dims,
            output_dims=10,
            mean_type='linear'
        )

        # 输出层
        last_layer = MyGPLayer(
            input_dims=hidden_layer2.output_dims,
            output_dims=output_dims,
            mean_type='constant'
        )

        super().__init__()

        self.hidden_layer1 = hidden_layer1
        self.hidden_layer2 = hidden_layer2
        self.last_layer = last_layer

        # 这里是从例子的 1 输出到多输出扩展的关键
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=output_dims)

    def forward(self, x):

        hidden_rep1 = self.hidden_layer1(x)
        hidden_rep2 = self.hidden_layer2(hidden_rep1)
        output = self.last_layer(hidden_rep2)

        return output

    def predict(self, test_x, test_y, batch_size, num_samples):
        """
        预测方法
        :param test_x:
        :param test_y:
        :param batch_size:
        :param num_samples: 在预测时，采样点的个数
        :return:
        """

        test_dataset = TensorDataset(test_x, test_y)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 输出的维数
        num_task = test_y.shape[1]

        with torch.no_grad():

            mean_pred = torch.zeros(1, num_task)
            mean_true = torch.zeros(1, num_task)

            for batch_x, batch_y in test_loader:

                with num_likelihood_samples(num_samples):

                    preds = self.likelihood(self(batch_x))

                    # preds.mean 的第一个维度会是 num_samples，我们先在该维度求个均值，把该维度消掉
                    mean_pred = torch.cat([mean_pred, preds.mean.mean(0).cpu()])
                    mean_true = torch.cat([mean_true, batch_y.cpu()])

        mean_pred = mean_pred[1:, :]
        mean_true = mean_true[1:, :]

        error = mean_pred - mean_true

        # 由于输出不止一个，所以评价指标分输出维度来计算
        MAE = torch.zeros(num_task, 1)
        MSE = torch.zeros(num_task, 1)
        RMSE = torch.zeros(num_task, 1)

        for i in range(num_task):
            MAE[i] = torch.mean(torch.abs(error[:, i]))
            MSE[i] = torch.mean(torch.pow(error[:, i], 2))
            RMSE[i] = torch.sqrt(MSE[i])

        return MAE, MSE, RMSE, mean_pred