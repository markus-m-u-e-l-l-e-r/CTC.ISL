import editdistance
import torch
from itertools import groupby
from constants import BLANK_ID


class Decoder(object):
    """
    Simple Decoder class that implements greedy decoding
    """

    def __init__(self, int_to_char):
        self.int_to_char = int_to_char

    def sequences_to_strings(self, sequences):
        strings = [self.sequence_to_string(x) for x in sequences]
        return strings

    def sequence_to_string(self, sequence):
        # we add a space between each unit, different decoders can overwrite this methods, e.g. to support bpe units
        return ' '.join([self.int_to_char[i] for i in sequence])

    def process_max_probs(self, max_probs, sizes=None):
        """
        Given a list of sequences, removes repeating symbols and the BLANK_ID
        e.g.: '0001110112220' -> '112'

        Arguments:
            max_probs: list of 1-d array of tensors
        """
        processed_strings = []
        assert sizes is None or len(max_probs) == len(sizes)
        if sizes is not None:
            for sequence, size in zip(max_probs, sizes.data):
                string = self.process_max_prob(sequence[:size])
                processed_strings.append(string)
        else:
            for sequence in max_probs:
                string = self.process_max_prob(sequence)
                processed_strings.append(string)

        return processed_strings

    def process_max_prob(self, sequence):
        # cast tensors to ints
        sequence = [x.item() for x in sequence]
        # remove repetitions
        sequence = [x[0] for x in groupby(sequence)]
        # filter blanks
        sequence = [x for x in sequence if x != BLANK_ID]
        return sequence

    def unflatten_ter(self, decoded_array, targets, target_sizes):
        assert len(decoded_array) == len(target_sizes)
        assert sum(target_sizes) == len(targets)
        # convert to ints
        targets = [x.item() for x in targets]
        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            assert len(targets) >= offset
            split_targets.append(targets[offset:offset + size])
            offset += size

        sum_ter = 0
        for s1, s2 in zip(decoded_array, split_targets):
            sum_ter += editdistance.eval(s1, s2)
        return sum_ter

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
        probs: Tensor of character probabilities, where probs[c,t]
            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        _, max_probs = torch.max(probs[1:,:].transpose(0, 1), 2)
        return self.process_max_probs(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes)
