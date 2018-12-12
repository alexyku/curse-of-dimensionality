# %%capture
import os
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import sys
sys.path.append('../src/mcmcp-bigan-generator-web-service/')
import time
from generate import floatX, load_params
from generate import _render_image as render
# utility functions for this notebook

from IPython.display import display
from IPython.display import clear_output
import ipywidgets as widgets

model_dir = '../src/mcmcp-bigan-generator-web-service/exp/imagenet_1000_size72_u-200_bigan/models'
load_params(model_dir, weight_prefix=None, resume_epoch=100)
print("WEIGHTS LOADED...")

def trunc_unif_categories(offset=0.4):
    # Sample `a` uniformly at random from the hypercube
    a = np.random.uniform(-1, 1, 200)
    # Samples `b` in a uniform square centered at `a`.
    # Redistributing probability mass to stay within the hypercube.
    b = np.array([np.random.uniform(
        max(-1, x - offset), min(1, x + offset)) for x in a])
    bayes = (0.5 * a + 0.5 * b)
    return (a, b, bayes)

def rand_categories(seed=None):
    if seed is not None:
        np.random.seed(seed)
    a = np.random.uniform(-1, 1, 200)
    b = np.random.uniform(-1, 1, 200)
    bayes = (0.5 * a + 0.5 * b)
    return (a, b, bayes)

def trunc_gauss_sample(mean, std):
    # Samples `x` from a truncated gaussian in the hypercube
    # centered at `mean` with standard deviation `std`.
    x = np.random.normal(mean, std)
    return np.minimum(1, np.maximum(-1, x))

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def get_config(offset=1.0,
               std=0.25,
               ndims=200,
               nblocks=5,
               ntrials=20,
               print_feedback=True,
               seed=None,
              ):
    # initialize an experiment
    config = dotdict()
    
    if seed is None:
        seed = np.random.randint(0, 9999)
    config.seed = seed
    np.random.seed(seed)
    
    # category creation
    config.offset = offset
    (config.bivimias, config.lorifens, config.bayes) = rand_categories(seed)
    config.dist = np.linalg.norm(
        config.bivimias - config.lorifens)
    
    # example sampling
    config.std = std
    config.ndims = ndims
    config.dims = sorted(np.random.choice(
        200, ndims, replace=False))
    config.bool_mask = np.zeros(200).astype(bool)
    config.bool_mask[config.dims] = True
    # config.scaled_std = std * (200 / ndims)

    # supervised or unsupervised?
    config.print_feedback = print_feedback
    
    # leaning/inference
    config.nblocks = nblocks
    config.ntrials = ntrials
    
    return config


# Mean sampling
def get_mean_example(config,
                     category='random',
                     render_img=True,
                    ):
    # generates an example latent vector
    # returns an example dict
    
    if category == 'random':
        category = np.random.choice(
            ['bivimias', 'lorifens'])
    # choose mean to use
    if category == 'bivimias':
        mean = config.bivimias
    elif category == 'lorifens':
        mean = config.lorifens
    noise = np.random.normal(0, config.std, 200)
    z = mean + (config.bool_mask * noise)
    z = np.tanh(z)
    example = {"category": category, "z": z}
    if render_img:
        example["img"] = render(z)
    return example


# Midpoint sampling
def get_example(config,
                category='random',
                render_img=True
               ):
    # generates an example latent vector
    # returns an example dict
    if category == 'random':
        category = np.random.choice(
            ['bivimias', 'lorifens'])
    # choose mean to use
    if category == 'bivimias':
        mean = config.bivimias
    elif category == 'lorifens':
        mean = config.lorifens
    z = config.bayes  # bayes optimal midpoint
    z[config.dims] = trunc_gauss_sample(
        mean, config.std)[config.dims]
    example = {"category": category, "z": z}
    if render_img:
        example["img"] = render(z)
    return example

def get_blocks(config):
    # returns a nested-list of example dicts
    np.random.seed(config.seed)
    return [
        [get_mean_example(config)for _ in range(config.ntrials)]
        for _ in range(config.nblocks)
    ]

def imshows(images,
            labels=None,
            n=None,
            axis='off',
            **kwargs
           ):
    # plots a list of images in a row
    n = len(images) if (n is None) else n
    for i in xrange(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i], **kwargs)
        plt.axis(axis)
        if labels: plt.title(labels[i])
            
def pilot_experiment(blocks, config, participant, logdir):
    config.start_time = time.time()
    
    status = dotdict()
    status.block_i = 0
    status.trial_i = 0
    status.question_i = 0
    status.ncorrect = 0

    # buttons
    start_button = widgets.Button(description="Start")
    continue_button = widgets.Button(description="Continue")
    bivimias_button = widgets.Button(description="Bivimiano")
    lorifens_button = widgets.Button(description="Lorifensky")

    def print_prefix(print_feedback=True):
        nquestions = config.nblocks * config.ntrials
        print "Question: %d/%d" % (status.question_i + 1, nquestions)
        print "Block: %d/%d" % (status.block_i + 1, config.nblocks)
        if config.print_feedback:
            if status.question_i == 0:
                pct_correct = 0.0
            else:
                pct_correct = status.ncorrect / float(status.question_i)
            print "Correct: %.4f" % pct_correct
        print

    def print_prompt():
        print("Is this a Bivimiano or a Lorifensky?")

    def is_done():
        nquestions = config.nblocks * config.ntrials
        return status.question_i == nquestions

    def get_trial():
        return blocks[status.block_i][status.trial_i]

    def display_image():
        plt.imshow(get_trial()["img"])
        plt.axis("off")
        plt.show()

    def log_and_increment(predicted_category):
        get_trial()[
            "predicted_category"
        ] = predicted_category
        status.ncorrect += int(
            predicted_category == get_trial()["category"])
        status.question_i += 1
        status.trial_i += 1
        if status.trial_i == config.ntrials:
            status.block_i += 1
            status.trial_i = 0

    def start_or_continue_click(_):
        clear_output()
        if not is_done():
            print_prefix()
            print_prompt()
            display_image()
            display(widgets.HBox(
                (bivimias_button,
                 lorifens_button)))
        else:
            config.end_time = time.time()
            
            print("Thank you for completing this experiment!")
            if status.question_i == 0:
                pct_correct = 0.0
            else:
                pct_correct = status.ncorrect / float(status.question_i)
            print "Correct: {:.1f}%".format(100 * pct_correct)
            print "\n"
            
            log_experiment(blocks, config, status, participant, logdir)

    def common_choice_click(category):
        clear_output()
        correct = get_trial()["category"]  # before update
        log_and_increment(category)
        print_prefix()
        if config.print_feedback:
            if correct == category:
                print "Correct!"
            else:
                print "Incorrect."
        else:
            print
        print "\n" * 14
        display(continue_button)

    start_button.on_click(start_or_continue_click)
    continue_button.on_click(start_or_continue_click)
    bivimias_button.on_click(lambda _: common_choice_click("bivimias"))
    lorifens_button.on_click(lambda _: common_choice_click("lorifens"))

    display(start_button)

def experiment_is_complete(blocks):
    return all([
        all(["predicted_category" in y for y in x])
        for x in blocks
    ])


def log_experiment(blocks, config, status, participant, logdir):
    if experiment_is_complete(blocks):
        start = format_timestamp(config.start_time)
        end = format_timestamp(config.end_time)
        
        participant_dir = 'participant_' + str(participant['id_num'])
        outdir = os.path.join(logdir, participant_dir, ( "ndims-" + str(config.ndims)
                                                       + "-std-" + str(config.std)
                                                       + "- " + start))
        os.makedirs(outdir)

        np.savez(os.path.join(outdir, "config.npz"), **config)
        np.savez(os.path.join(outdir, "blocks.npz"), *blocks)

        with open(os.path.join(outdir, 'log.txt'), 'w') as f:
            f.write('Participant: {} ({})'.format(participant['id_num'], participant['initials']) + '\n')
            f.write('Start time: {}'.format(start) + '\n')
            f.write('End time: {}'.format(end) + '\n')
            f.write('Elapsed time: {:.2f}s'.format(config.end_time - config.start_time) + '\n')
            f.write('Dims: {}'.format(config.ndims) + '\n')
            f.write('Std: {}'.format(config.std) + '\n')
            f.write('Correct: {} / {}'.format(status.ncorrect, status.question_i) + '\n')
            if status.question_i > 0:
                f.write('Percent correct: {:.4f}'.format(status.ncorrect / status.question_i) + '\n')

        print("Logged experiment to %s" % outdir)
    else:
        print("Experiment is incomplete, not logging.")
        
        
def format_timestamp(t):
    return time.strftime('%c', time.localtime(t))