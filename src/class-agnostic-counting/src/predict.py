import argparse
import math
import os

import numpy as np
from matplotlib import pyplot as plt

# ===========================================
#        Parse the argument
# ===========================================

# essential arguments for predicting
parser = argparse.ArgumentParser()
parser.add_argument('--ex_img', type=str, required=True, help="the object image")
parser.add_argument("--in_img_dir", type=str, required=True, help="a folder containing only the input images")
parser.add_argument('--dataset', choices=['flowers1', 'flowers2'],
                    type=str, help='test on which specific dataset.', required=True)
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--gmn_path', type=str, required=True, help='pre-trained model path')
parser.add_argument('--output_path', type=str, default="./output", help='the output will be saved here')

parser.add_argument('--net', default='resnet50', choices=['resnet50'], type=str)
parser.add_argument('--optimizer', default='adam', choices=['adam'], type=str)
parser.add_argument('--mode', default='adapt', choices=['pretrain', 'adapt'], type=str,
                    help='pretrain on tracking data or adapt to specific dataset.')
parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--warmup_ratio', default=0, type=float)
parser.add_argument('--batch_size', default=24, type=int)
parser.add_argument('--epochs', default=36, type=int,
                    help='number of total epochs to run')
parser.add_argument('--data_path', default='', type=str)

global args
args = parser.parse_args()

global trn_config

import utils as ut

# ==> gpu configuration
ut.initialize_GPU(args)


# args for loading the model
# see main.py for more details regarding each argument here
# class Args:
#     gpu = "1"
#     net = "resnet50"
#     optimizer = "adam"
#     mode = "adapt"
#     dataset = "flowers2"
#     lr = 0.0005
#     warmup_ratio = 0.0
#     resume = "models/flowers2.h5"
#     batch_size = 5
#     epochs = 36
#     gmn_path = "models/flowers2.h5"
#     data_path = "data/flowers/"
#
#
# args = Args()

def get_model():
    # # ==> import library
    import data_loader as data_loader
    import model_factory as model_factory

    # ==> get dataset information
    global trn_config
    trn_config = data_loader.get_config(args)

    # ==> load networks
    # well that adapt argument gotta be true otherwise you won't be able to load the weights
    gmn = model_factory.two_stream_matching_networks(trn_config, sync=False, adapt=True)
    gmn.load_weights(args.gmn_path, by_name=True)

    # ==> print model summary
    # gmn.summary()

    return gmn


def save_result(name, out):
    out = np.squeeze(out, axis=0)
    out = np.squeeze(out, axis=2)
    number_of_flowers = math.ceil(np.sum(out / 100))
    plt.imsave(os.path.join(args.output_path, args.dataset + "_" + str(number_of_flowers) + "_" + name), out)


if __name__ == '__main__':
    import data_generator as data_generator

    # load model
    model = get_model()

    # load object image
    ex_patch = ut.load_data(args.ex_img, dims=trn_config.patchdims, pad=trn_config.pad)
    # preprocess object image
    ex_patch = np.expand_dims(ex_patch, axis=0)
    ex_patch = data_generator.preprocess_input(np.array(ex_patch, dtype='float32'))

    # iterate over each image in the input image folder and predict the result
    input_images_list = os.listdir(args.in_img_dir)
    for input_img_name in input_images_list:
        print("predicting for %s " % input_img_name)
        input_img = ut.load_data(os.path.join(args.in_img_dir, input_img_name), dims=trn_config.imgdims,
                                 pad=trn_config.pad)

        # pre-processing for the input image
        input_img = np.expand_dims(input_img, axis=0)
        input_img = data_generator.preprocess_input(np.array(input_img, dtype='float32'))

        # input for the model
        inputs = {'image_patch': ex_patch, 'image': input_img}

        # predict
        outputs = model.predict(inputs, batch_size=1)

        # save the model
        save_result(input_img_name, outputs)
