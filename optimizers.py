import torch.optim as optim

def get_optimizer(optimizer_name, model_params, lr=0.001):
    if optimizer_name == "adam":
        return optim.Adam(model_params, lr=lr)
    elif optimizer_name == "sgd":
        return optim.SGD(model_params, lr=lr)
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model_params, lr=lr)
    else:
        raise ValueError("Unknown optimizer: " + optimizer_name)
