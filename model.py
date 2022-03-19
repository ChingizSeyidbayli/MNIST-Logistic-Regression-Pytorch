import torch
import torch.nn as nn

# Create Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward_propagation(self, x):
        out = self.linear(x)
        return out

input_dim = 28*28 # size of input image
output_dim = 10  # labels of output ==> 1,2,3,4,5,6,7,8,9,10

#logistic regression model
model = LogisticRegressionModel(input_dim, output_dim)

# Loss - CrossEntropy
error = nn.CrossEntropyLoss()

# SGD(Stochastic Gradient Descent) Optimizer 
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)