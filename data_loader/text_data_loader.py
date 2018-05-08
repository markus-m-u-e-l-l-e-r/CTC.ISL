from torch.utils.data import Dataset, DataLoader
import torch
import h5py
import random
import numpy
import util
from collections import namedtuple

UTTERANCE = namedtuple('UTTERANCE', ['spk_id', 'start_frame', 'end_frame', 'targets'], verbose=False)


class SequenceDataset(Dataset):
    def __init__(self, id_to_unit, common_args, custom_args):
        self.audio_file = h5py.File(custom_args["audio_features"], 'r')

        self.id_to_unit = id_to_unit
        self.unit_to_id = {v: k for k, v in id_to_unit.items()}

        self.start_idx = -1
        if "start_idx" in custom_args:
            self.start_idx = custom_args["start_idx"]

        self.frame_shift = float(common_args["frame_shift"]) # in seconds
        self.step_size = int(common_args["step_size"])
        self.left = int(common_args["context"])
        self.right = int(common_args["context"])

        self.utterances = self._read_text(custom_args["input_text"], self.unit_to_id)

    def _filter_function(self, utt):
        h5_path = utt.spk_id
        if h5_path in self.audio_file and utt.start_frame < len(self.audio_file[h5_path]) \
                and len(utt.targets) <= (utt.end_frame - utt.start_frame - self.step_size) // self.step_size:
            return True
        # just to give more information
        else:
            if h5_path not in self.audio_file:
                print("The following h5_path was not found: {}".format(h5_path))
            elif not len(utt.targets) <= (utt.end_frame - utt.start_frame - self.step_size) // self.step_size:
                print("Ignoring {}, targets are longer than input sequence".format(h5_path))
            else:
                print("WARN: Ignoring {}, frames audio {}, start frame {}"
                      .format(utt.spk_id, len(self.audio_file[h5_path]), utt.start_frame))
        return False

    def _read_text(self, file_path, unit_to_id):
        """
        processes a given text file
        :param file_path:
        :param unit_to_id:
        :return: list of UTTERANCES sorted (asc) by their length, unusable utterances are already filtered out
        """
        utterances = []
        with open(file_path, mode="r") as file:
            for line in file:
                split = line.split()
                assert len(split) >= 3 and \
                    "The transcription file should have the following format <speaker_id> <start_time> <end_time>"
                # we skip utterances with no transcriptions
                if len(split) == 3:
                    continue
                
                spk_id, start_time, target_units = split[0], float(split[1]), split[3:]

                # end time = x means that we do not know an end time, so we use the last frame of the audio data
                if split[2] == "x":
                    end_frame = len(self.audio_file[spk_id])
                else:
                    end_time = float(split[2])
                    end_frame = int(end_time / self.frame_shift) + 1

                start_frame = int(start_time / self.frame_shift)

                target_ids = [unit_to_id.get(x, 0) for x in target_units]
                utt = UTTERANCE(spk_id, start_frame, end_frame, target_ids)
                utterances.append(utt)
        original_number = len(utterances)
        # use filter function to exclude mismatches between audio file and transcription file
        utterances = list(filter(self._filter_function, utterances))
        # sort by length
        utterances.sort(key=lambda k: k.end_frame - k.start_frame)
        print("Keeping {} utterances out of {} utterances".format(len(utterances), original_number))
        return utterances

    def __len__(self):
        return len(self.utterances)

    def _feat_add_context(self, feat):
        feature_list = []
        for shift in range(-self.left, self.right+1):
            feature_list.append(numpy.roll(feat, shift, axis=0))
        return numpy.concatenate(feature_list, axis=1)

    def get_utt_info(self, utt_id):
        utt = self.utterances[utt_id]
        filename = utt.spk_id
        channel = '0'
        units = [self.id_to_unit[token] for token in utt.targets]
        return {"start_time": utt.start_frame * self.frame_shift,
                "frame_rate": self.frame_shift * self.step_size,
                "filename": filename,
                "channel": channel,
                "reference": units
                }

    def __getitem__(self, index):
        utt = self.utterances[index]
        h5_path = utt.spk_id
        # randomly select start_idx, add self.left to circumvent edge cases
        offset = self.start_idx
        if offset < 0:
            offset = random.randrange(self.step_size)

        # calculate ids
        start_idx = utt.start_frame + offset
        end_idx = start_idx + self._get_item_frames(index)
        assert start_idx < len(self.audio_file[h5_path])

        np_feats = self._feat_add_context(self.audio_file[h5_path][start_idx:end_idx])
        #subsample
        features = torch.from_numpy(np_feats[::self.step_size])
        assert features.shape[0] >= len(utt.targets)
        return features, utt.targets, index

    # length before subsampling
    def _get_item_frames(self, index):
        utt = self.utterances[index]
        length = utt.end_frame - utt.start_frame - self.step_size
        return length

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
