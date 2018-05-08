import torch


class SGD:
    def __init__(self, model, config):
        self.performances = [float("inf")]
        self.lr = float(config["learning_rate"])
        self.momentum = float(config['momentum'])
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, self.momentum, nesterov=True)
        self.threshold = 0.1

    def new_epoch(self, performance):
        if (performance + self.threshold) > self.performances[-1]:
            # reset, half learning rate
            self.lr *= 0.5
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, self.momentum, nesterov=True)
            print("Performance did not improve by more than {}; old performance: {}, new performance: {}".format(self.threshold, self.performances[-1], performance))
            print("INFO: halving learning rate, new learning rate: {}".format(self.lr))
        self.performances.append(performance)

    def get_optimizer(self):
        return self.optimizer


class Adam:
    def __init__(self, model, config):
        self.performances = [float("inf")]
        self.lr = float(config["learning_rate"])
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.threshold = 0.1

    def new_epoch(self, performance):

        if (performance + self.threshold) > self.performances[-1]:
            # reset, half learning rate
            self.lr *= 0.5
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
            print("Performance did not improve by more than {}; old performance: {}, new performance: {}".format(self.threshold, self.performances[-1], performance))
            print("INFO: halving learning rate, new learning rate: {}".format(self.lr))
        self.performances.append(performance)

    def get_optimizer(self):
        return self.optimizer
