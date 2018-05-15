from torch.utils.data import Dataset, DataLoader
import torch
import h5py
import random
import numpy
import util
from collections import namedtuple

UTTERANCE = namedtuple('UTTERANCE', ['id', 'targets'], verbose=False)


class SequenceDataset(Dataset):
    def __init__(self, id_to_unit, common_args, custom_args):
        self.audio_file = h5py.File(custom_args["audio_features"], 'r')

        self.id_to_unit = id_to_unit
        self.unit_to_id = {v: k for k, v in id_to_unit.items()}

        self.start_idx = -1
        if "start_idx" in custom_args:
            self.start_idx = custom_args["start_idx"]

        self.step_size = int(common_args["step_size"])
        self.left = int(common_args["context"])
        self.right = int(common_args["context"])

        self.utterances = self._read_text(custom_args["input_text"], self.unit_to_id)

    def _read_text(self, file_path, unit_to_id):
        """
        processes a given text file
        :param file_path:
        :param unit_to_id:
        :return: list of UTTERANCES sorted (asc) by their length, unusable utterances are already filtered out
        """
        utterances = []
        with open(file_path, mode="r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                target_units = line.split()
                # we skip utterances with no transcriptions
                if len(target_units) == 0:
                    continue
                
                target_ids = [unit_to_id[x] for x in target_units]
                utt = UTTERANCE(str(i), target_ids)
                utterances.append(utt)
        # sort by length
        utterances.sort(key=lambda k: len(self.audio_file[k.id]))
        return utterances

    def __len__(self):
        return len(self.utterances)

    def get_utt_id(self, utt_id):
        utt = self.utterances[utt_id]
        return utt.id

    def _feat_add_context(self, feat):
        feature_list = []
        for shift in range(-self.left, self.right+1):
            feature_list.append(numpy.roll(feat, shift, axis=0))
        return numpy.concatenate(feature_list, axis=1)

    def __getitem__(self, index):
        utt = self.utterances[index]
        h5_path = utt.id
        # randomly select start_idx, add self.left to circumvent edge cases
        start_idx = self.start_idx if self.start_idx >= 0 else random.randrange(self.step_size)

        np_feats = self._feat_add_context(self.audio_file[h5_path][start_idx:])
        #subsample
        features = torch.from_numpy(np_feats[::self.step_size])
        assert features.shape[0] >= len(utt.targets)
        return features, utt.targets, index

    # length before subsampling
    def _get_item_frames(self, index):
        utt = self.utterances[index]
        return len(self.audio_file[utt.id])

    # after subsampling
    def get_item_length(self, index):
        return self._get_item_frames(index) // self.step_size


class BatchSampler(object):
    """
    Args: 
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, dataset, batch_size, shuffle):
        self.batch_size = batch_size
        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.shuffle = shuffle
        self.batches = self._create_batches()
        self.length = len(self.batches)

    def _create_batches(self):
        batches = []
        current_id = 0

        # create batches of equal length
        while current_id < self.dataset_size:
            #
            batch = [current_id]
            current_length = self.dataset.get_item_length(current_id)

            # add everything of the same length
            for i in range(1, self.batch_size):
                id_to_add = current_id + i
                if id_to_add < self.dataset_size and current_length == self.dataset.get_item_length(id_to_add):
                    batch.append(id_to_add)

            assert 1 <= len(batch) <= self.batch_size
            batches.append(batch)
            current_id = sum([len(x) for x in batches])
        return batches

    def __iter__(self):
        indices = torch.LongTensor(range(self.length))
        if self.shuffle:
            indices = iter(torch.randperm(self.length).long())
        for idx in indices:
            yield self.batches[idx]

    def __len__(self):
        return self.length


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        assert len(args) == 1
        dataset = args[0]

        # this is ugly, but we basically handle the named arguments ourself
        common_args = kwargs.pop('common_args')
        custom_args = kwargs.pop('custom_args')
        batch_size = kwargs.pop('batch_size')
        shuffle = kwargs.pop('shuffle', False)
        # set collate function
        collate = util.import_func(common_args["collate_fn"])
        batch_sampler = BatchSampler(dataset, batch_size, shuffle)
        kwargs.update(collate_fn=collate)
        kwargs.update(batch_sampler=batch_sampler)
        # call super init with new arguments 
        super(AudioDataLoader, self).__init__(*args, **kwargs)
