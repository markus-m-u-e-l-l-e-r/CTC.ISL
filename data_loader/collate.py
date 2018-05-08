import torch


def collate_3D(batch):
    """ processes and combines a 3D batch to produce Variables needed for the CTC function
    :param batch: list of tuples (inputs, targets, utt_ids)
        expected shape tensor: (seq_length x feature_dim)
    :returns: tuple (utt_ids, input
    batch: list of tuples (scores, targets, scores_sizes, targets_sizes, utt_ids)
        inputs: tensor of shape: (out_max_seq_length x batch_size x feature_dim)
        targets: 1D tensor containing all the targets
        input_percentages: tensor specifying the sequence relative length of each input element
        target_sizes: tensor specifying the length of each target sequence
        utt_ids: list of utterance ids used in this batch
    """

    # get longest sample by sequence length
    longest_sample = max(batch, key=lambda x: x[0].size(0))[0]
    max_seq_length, feature_size = longest_sample.size()
    minibatch_size = len(batch)

    # 3D tensor
    inputs = torch.zeros(max_seq_length, minibatch_size, feature_size)
    input_sizes = torch.IntTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)

    targets = []
    h5_paths = []
    for idx, (tensor, target, path) in enumerate(batch):
        h5_paths.append(path)
        seq_length = tensor.size(0)
        inputs[:seq_length,idx,:].copy_(tensor)
        input_sizes[idx] = seq_length
        target_sizes[idx] = len(target)
        targets.extend(target)

    targets = torch.IntTensor(targets)
    return inputs, targets, input_sizes, target_sizes, h5_paths

