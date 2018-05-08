import torch


class BaseModel(torch.nn.Module):
    """
    Base Model that defines the requirements of the models and
    provides helper functions
    """

    def forward(self, x):
        """ Performs a forward pass of the network .
        :param x: Input Variable, can either be 3D for RNNs/ or 4D for CNNs
            expected shape 3D: (seq_length x batch_size x feature_dim)
        :returns: Variable of shape (seq_length x batch_size x feature_dim)
        """
        raise NotImplementedError

    def save(self, path):
        package = self.model_args
        package["state_dict"] = self.state_dict()
        torch.save(package, path)

    def _load_state_dict_from_file(self, path, map_to_cpu=False, remove_params=None):
        if map_to_cpu:
            p = torch.load(path, map_location=lambda storage, loc: storage)
        else:
            p = torch.load(path)

        state_dict = p.pop("state_dict")

        # Optionally remove not needed parameters from state dict
        if remove_params is not None:
            for param in remove_params.split(','):
                del state_dict[param]
        print("Load state dict from {}".format(path))
        self.load_state_dict(state_dict, strict=False)
