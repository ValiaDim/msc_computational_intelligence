from __future__ import print_function

import collections
import inspect
import os
import re
import warnings
import glob

import numpy as np
from PIL import Image


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_folders_for_training(options):
    train_experiment_name = options.train_experiment_name
    root_log = "trainings"
    mkdir(root_log)
    if train_experiment_name:
        training_dir = os.path.join(root_log, train_experiment_name)
        dir_no = 0
        changed_dir_name = False
        while True:
            dir_no += 1
            if os.path.isdir(training_dir):
                training_dir = os.path.join(root_log, train_experiment_name + "_" + str(dir_no))
                changed_dir_name = True
            else:
                break
        if changed_dir_name:
            print("The experiment name you provided already exists. Saving under {}".format(training_dir))
    else:
        training_dir = os.path.join(root_log, "training")
        dir_no = 0
        while True:
            dir_no += 1
            if os.path.isdir(training_dir):
                training_dir = os.path.join(root_log, "training_" + str(dir_no))
            else:
                break
        print("You didn't provide any experiment name. Saving under {}".format(training_dir))
    mkdir(training_dir)
    log_folder = os.path.join(training_dir, 'logs')
    checkpoint_folder = os.path.join(training_dir, 'checkpoints')
    plot_folder = os.path.join(training_dir, 'plots')
    mkdirs([log_folder, checkpoint_folder, plot_folder])
    # log the options of this experiment
    args = vars(options)
    # save to the disk
    file_name = os.path.join(log_folder, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
    return training_dir


def logger(message, dir, change_classifier=True):
    log_path = os.path.join(dir, "logger")
    f = open(log_path, "a")
    f.write(message)
    if change_classifier:
        f.write("--------------------------------------------------------------------\n")
    f.close()



