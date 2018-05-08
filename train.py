import argparse
import json
import yaml
import random

import torch
from warpctc_pytorch import CTCLoss
from decoder import Decoder
import util


def main():
    parser = argparse.ArgumentParser(description='CTC Training')
    parser.add_argument('--config', help='YAML config', default=None)
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

    if "seed" in config["general"]:
        random.seed(config["general"]["seed"])

    logger = util.Logger(config['general'].get("logfile", "train.log"))
    ctc_loss = CTCLoss()

    # Load labels
    with open(config["general"]["labels"], 'r') as label_file:
        labels_tmp = json.load(label_file)
    labels = {int(key): labels_tmp[key] for key in labels_tmp}

    # Init data set
    trainDS = util.import_func(config['data']['train']['data_set'])
    validDS = util.import_func(config['data']['valid']['data_set'])
    train_dataset = trainDS(labels, config['data']['common'], config['data']['train'])
    valid_dataset = validDS(labels, config['data']['common'], config['data']['valid'])

    # Init data loader
    trainLD = util.import_func(config['data']['train']['data_loader'])
    validLD = util.import_func(config['data']['valid']['data_loader'])
    train_shuffle = config['data']['train']['shuffle'] if 'shuffle' in config['data']['train'] else config['data']['common'].get('shuffle', False)

    train_loader = trainLD(train_dataset,
                           common_args=config['data']['common'],
                           custom_args=config['data']['train'],
                           batch_size=config['data']['common']['batch_size'],
                           num_workers=config['data']['common']['num_workers'],
                           shuffle=train_shuffle)
    valid_loader = validLD(valid_dataset,
                           common_args=config['data']['common'],
                           custom_args=config['data']['valid'],
                           batch_size=config['data']['common']['batch_size'],
                           num_workers=config['data']['common']['num_workers'])

    feat_size = config['data']['common']['input_dim'] * (2 * config['data']['common']['context'] + 1)

    model_class = util.import_func(config['model']['class'])

    config['model']['feat_size'] = feat_size
    config['model']['num_classes'] = len(labels)
    model = model_class(config['model'])

    # torch.device object used throughout this script
    device = torch.device("cuda" if config['general'].get("use_cuda", False) else "cpu")
    model = torch.nn.DataParallel(model).to(device)
    print(model)

    decoder = Decoder(labels)

    # Choose SGD as default optimizer if no optimizer is defined
    optimizer_class = util.import_func(config["general"].get("optimizer", "optimizer.SGD"))

    optimizer_object = optimizer_class(model, config["general"])
    logger.log("Start training")

    def forward_pass(data):
        inputs, targets, input_sizes, target_sizes, h5_paths = data
        inputs = inputs.to(device)

        # out.shape = (TxNxH)
        out = model(inputs)
        loss = ctc_loss(out, targets, input_sizes, target_sizes)

        # computer ler
        decoded_output = decoder.decode(out.data, input_sizes)  # Contents: array of arrays with idXs

        errors = float(decoder.unflatten_ter(decoded_output, targets.data, target_sizes.data))
        examples = out.size(1)
        targets = len(targets)
        return loss, examples, errors, targets

    for epoch in range(config['general']['epochs']):
        # TRAINING
        model.train(True)
        train_info = util.EvalInfo()

        for i, (data) in enumerate(train_loader):
            loss, batch_size, number_errors, number_targets = forward_pass(data)
            loss_value = loss.item()

            # compute gradient
            optimizer = optimizer_object.get_optimizer()
            optimizer.zero_grad()
            if loss_value != float("inf") and loss_value != -float("inf"):
                loss.backward()
                train_info.update(loss_value, batch_size, number_errors, number_targets)
            else:
                print("Loss was infinity")

            torch.nn.utils.clip_grad_norm_(model.parameters(), config['general'].get('max_norm', 5.0))
            optimizer.step()
        logger.log(train_info.get_info(name="Training", epoch=epoch))

        # VALIDATION
        model.train(False)
        valid_info = util.EvalInfo()

        with torch.no_grad():
            for i, (data) in enumerate(valid_loader):
                loss, batch_size, number_errors, number_targets = forward_pass(data)
                valid_info.update(loss.item(), batch_size, number_errors, number_targets)
        logger.log(valid_info.get_info(name="Validation", epoch=epoch))

        # Update Optimizer and save model
        optimizer_object.new_epoch(valid_info.avg_ter())
        model.module.save(config['general']['final_dump'])

if __name__ == '__main__':
    main()
