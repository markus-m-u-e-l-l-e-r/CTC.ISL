import argparse
import json
import yaml
import random

import h5py
import torch
from decoder import Decoder
import util


def main():
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument('config', help='YAML config')
    parser.add_argument('--hyp_file', default=None, help="filename to write greedy hypothesis to")
    parser.add_argument('--logits_file', default=None, help="h5 filename to write logits to")
    args = parser.parse_args()

    print("Using config {}".format(args.config))

    assert args.config is not None and "No configfile specified, aborting."

    print("Torch version: {}".format(torch.__version__))
    try:
        print("Using CuDNN: {}".format(torch.backends.cudnn.version()))
    except:
        print("Not using CuDNN")

    with open(args.config, 'r') as inFP:
        config = yaml.load(inFP, Loader=yaml.Loader)
        inFP.seek(0)
        for line in inFP:
            print(line.strip('\n'))

    if 'seed' in config['general']:
        random.seed(config['general']['seed'])

    # Load labels
    with open(config['general']['labels'], 'r') as label_file:
        labels_tmp = json.load(label_file)
    labels = {int(key): labels_tmp[key] for key in labels_tmp}

    # Init data set
    testDS = util.import_func(config['data']['valid']['data_set'])
    test_dataset = testDS(labels, config['data']['common'], config['data']['valid'])

    # Init data loader
    testLD = util.import_func(config['data']['valid']['data_loader'])
    test_loader = testLD(test_dataset,
                           common_args=config['data']['common'],
                           custom_args=config['data']['valid'],
                           batch_size=config['data']['common']['batch_size'],
                           num_workers=config['data']['common']['num_workers'])

    feat_size = config['data']['common']['input_dim'] * (2 * config['data']['common']['context'] + 1)

    model_class = util.import_func(config['model']['class'])

    config['model']['feat_size'] = feat_size
    config['model']['num_classes'] = len(labels)
    model = model_class(config['model'])

    device = torch.device("cuda" if config['general'].get("use_cuda", False) else "cpu")
    model = torch.nn.DataParallel(model).to(device)
    print(model)

    decoder = Decoder(labels)

    def forward_pass(inputs):
        inputs, targets, input_sizes, target_sizes, utt_ids = data
        inputs = inputs.to(device)

        # out.shape = (TxNxH)
        logits = model(inputs)

        # computer ler
        decoded_output = decoder.decode(logits.data)  # Contents: array of arrays with idXs
        return decoded_output, utt_ids, logits.data

    # open files
    logits_file = None
    hyp_file = None

    if args.logits_file:
        logits_file = h5py.File(args.logits_file, "a")

    if args.hyp_file:
        hyp_file = open(args.hyp_file, "w")

    # Decoding
    model.train(False)
    with torch.no_grad():
        for data in test_loader:
            decoded_outputs, utt_ids, logits = forward_pass(data)

            for i, (decoded_output, utt_id) in enumerate(zip(decoded_outputs, utt_ids)):
                decoded_string = " ".join([labels[x] for x in decoded_output]).replace("@@ ", "")
                utt = test_dataset.get_item_info(utt_id)
                # (spk_id='blair4', start_frame=6375, end_frame=8350)
                if hyp_file is not None:
                    hyp_file.write("{} {} {} {}\n".format(utt.spk_id, utt.start_frame, utt.end_frame, decoded_string))
                utt_name = "{} {}\n".format(utt.spk_id, utt.start_frame, utt.end_frame)
                if logits_file is not None and utt_name not in logits_file:
                    logits_file.create_dataset(utt_name, data=logits[:,i,:])

    # Cleanup
    if logits_file is not None:
        logits_file.close()

    if hyp_file is not None:
        hyp_file.close()

if __name__ == '__main__':
    main()
