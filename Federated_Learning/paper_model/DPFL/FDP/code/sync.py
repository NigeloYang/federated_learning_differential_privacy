import torch


def run_ma(global_model, local_models, selected, device, alpha=0.1):
    """ Elastic Model Averaging """

    if isinstance(alpha, float):
        alpha = [alpha] * len(local_models)
    else:
        assert len(alpha) == len(local_models)

    with torch.no_grad():

        weight = 1.0 / len(selected)

        # sum up parameters
        param_sum = {}
        for ii in selected:
            for name, param in local_models[ii].state_dict().items():
                if name not in param_sum:
                    param_sum[name] = torch.zeros_like(param)
                param_sum[name].add_(param * weight)

        # update global model
        for name, param in global_model.state_dict().items():
            param.add_(param_sum[name] - param)

        # elastic averaging
        for ii in selected:
            for name, param in local_models[ii].state_dict().items():
                param.add_(alpha[ii] * (param_sum[name] - param))
