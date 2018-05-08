from torch import nn
from .base_model import BaseModel


class LSTM(BaseModel):
    def __init__(self, model_args):
        super(LSTM, self).__init__()

        self.model_args = model_args
        self.directions = 2 if model_args['bidirectional'] else 1
        self.dropout = model_args.get('dropout', False)

        # define weights and layers
        self.lstm = nn.LSTM(model_args['feat_size'], model_args['rnn_hidden_size'], model_args['rnn_layers'], True, False,
                            self.dropout, model_args['bidirectional'])
        self.linear = nn.Linear(model_args['rnn_hidden_size'] * self.directions, model_args['num_classes'], bias=False)
        if "load" in model_args:
            self._load_state_dict_from_file(model_args["load"],
                                            model_args.get("map_to_cpu", False),
                                            model_args.get("remove_params", None))

    def forward(self, x):
        """
        expected x shape: (seq_length, batch_size, input_features)
        """
        y, _ = self.lstm(x)
        seq_length, batch_size, features = y.size()
        y = y.view(-1, features)
        scores = self.linear(y)
        scores = scores.view(seq_length, batch_size, -1)
        return scores

