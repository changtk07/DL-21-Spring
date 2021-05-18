import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        activation = dict(
            relu     = torch.nn.functional.relu,
            sigmoid  = torch.sigmoid,
            identity = lambda x : x
        )
        
        f = activation[ self.f_function ]
        g = activation[ self.g_function ]
            
        z1 = torch.mm(self.parameters['W1'], x.t()) + self.parameters['b1'].unsqueeze(1)
        z2 = f(z1);
        z3 = torch.mm(self.parameters['W2'], z2) + self.parameters['b2'].unsqueeze(1)
        y_hat = g(z3)
        
        self.cache['x'] = x
        self.cache['z1'] = z1
        self.cache['z2'] = z2
        self.cache['z3'] = z3
        
        return y_hat.t()
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        derivative = dict(
            relu     = lambda x : 1. * (x > 0.),
            sigmoid  = lambda x : torch.exp(-x) / (1.+torch.exp(-x))**2,
            identity = lambda x : 1.
        )
        
        df = derivative[ self.f_function ]
        dg = derivative[ self.g_function ]
        
        dJdb2 = dJdy_hat * dg(self.cache['z3'].t())
        dJdW2 = torch.mm(dJdb2.t(), self.cache['z2'].t())
        dJdz2 = torch.mm(dJdb2, self.parameters['W2'])
        dJdb1 = dJdz2 * df(self.cache['z1'].t())
        dJdW1 = torch.mm(dJdb1.t(), self.cache['x'])
        
        self.grads['dJdb2'] = torch.sum(dJdb2, 0)
        self.grads['dJdW2'] = dJdW2
        self.grads['dJdb1'] = torch.sum(dJdb1, 0)
        self.grads['dJdW1'] = dJdW1
    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    J = torch.nn.functional.mse_loss(y_hat, y)
    n = torch.numel(y)
    dJdy_hat = 2./n * (y_hat - y)
    
    return J, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    eps = 1.e-12
    J = torch.nn.functional.binary_cross_entropy(y_hat, y)
    n = torch.numel(y)
    dJdy_hat = -1./n * (y/(y_hat+eps) - (1.-y)/(1.-y_hat+eps))

    return J, dJdy_hat
