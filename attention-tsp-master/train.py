import os
import time
from tqdm import tqdm
import torch
import math
# import _pickle as pickle
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from tensorboard_logger import log_value

from attention_model import set_decode_type
from log_utils import log_values
import json
import numpy as np



def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def make_var(val, cuda=False, **kwargs):
    var = Variable(val, **kwargs)
    if cuda:
        var = var.cuda()
    return var


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        cost, log_p, pi, _ = model(make_var(bat, opts.use_cuda, volatile=True))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))

    termijn = open("termijnbewaking/{}.csv".format(opts.run_name), "a")
    termijn.write(str(epoch)+","+str(time.ctime())+"\n")
    termijn.close()

    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()
    lr_scheduler.step(epoch)

    if not opts.no_tensorboard:
        log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = 0
    if (opts.experiment == "supervised"):

        exp_params = loadArgument(opts.supervised_parameter)["data"]
        
        correct_tuple = 0
        counter = -1

        while (counter<epoch):
            for _ in range(exp_params[correct_tuple][1]):

                counter += 1
                if (counter == epoch):
                    correct_tuple = tuple(exp_params[correct_tuple])
                    break
            if (type(correct_tuple) == type(tuple([1,2,3]))):
                break
            correct_tuple += 1

        training_dataset = baseline.wrap_dataset(problem.make_dataset(size=correct_tuple[0], num_samples=opts.epoch_size, entropy=correct_tuple[2], target=opts.graph_size))
        # training_dataset = baseline.wrap_dataset(problem.make_dataset(size=correct_tuple[0], num_samples=1, entropy=correct_tuple[2]))

    elif (opts.experiment == "adaptive"):

        adaptief_vorige_score, adaptief_current_graph_size, adaptief_current_entropy = loadAdaptief(opts)
                        
        if (not len(adaptief_vorige_score) >=  (opts.epoch_size/opts.batch_size)*2):

            if (len(adaptief_vorige_score) <= (opts.epoch_size/opts.batch_size) ):
                print ("1/2 initial datasets of graph size")
            else: 
                print ("2/2 initial datasets of graph size")
            training_dataset = baseline.wrap_dataset(problem.make_dataset(size=adaptief_current_graph_size, num_samples=opts.epoch_size, entropy=adaptief_current_entropy, target=opts.graph_size))
        elif (adaptief_current_graph_size >= opts.graph_size and adaptief_current_entropy >= 1.0):
            training_dataset = baseline.wrap_dataset(problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size, entropy=1.0, target=opts.graph_size))
        else:
            exp_params = loadArgument(opts.adaptive_parameter)["data"]
            percentage = exp_params[0]
            stepsize_size = exp_params[1]
            stepsize_entropy = exp_params[2]
            half = int(len(adaptief_vorige_score)/2)
            nieuw = np.mean(adaptief_vorige_score[half:])
            oud = np.mean(adaptief_vorige_score[:half])
            print("average score before {}, average score after {}".format(oud, nieuw))

            improvement = -((nieuw-oud)/oud)

            if (improvement < percentage):
 
                adaptief_current_entropy += stepsize_entropy
                adaptief_current_graph_size += stepsize_size
                adaptief_vorige_score.clear()
                print ("improvement {} was lower than percentage {}".format(improvement, percentage))
                print ("dataset upped to: entropy {}, size {}".format(adaptief_current_entropy, adaptief_current_graph_size))
            else:
                print ("improvement {} was higher than percentage {}".format(improvement, percentage))
            training_dataset = baseline.wrap_dataset(problem.make_dataset(size=adaptief_current_graph_size, num_samples=opts.epoch_size, entropy=adaptief_current_entropy, target=opts.graph_size))
        saveAdaptief(adaptief_vorige_score, adaptief_current_graph_size, adaptief_current_entropy, opts)
        fn = open("termijnbewaking/{}_F_n.csv".format(opts.run_name), "a")
        fn.write(str(epoch) + "," + str(adaptief_current_graph_size) + "," + str(adaptief_current_entropy) + "\n")
        fn.close()
    elif (opts.experiment == "unsupervised"):
        pass
    else:
        training_dataset = baseline.wrap_dataset(problem.make_dataset(size=opts.graph_size, num_samples=opts.epoch_size, entropy=1.1, target=opts.graph_size))

    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    print('Saving model and state...')
    torch.save(
        {
            'model': get_inner_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
            'baseline': baseline.state_dict()
        },
        os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
    )

    avg_reward = validate(model, val_dataset, opts)

    if not opts.no_tensorboard:
        log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        opts
):

    x, bl_val = baseline.unwrap_batch(batch)
    x = make_var(x, opts.use_cuda)
    bl_val = make_var(bl_val, opts.use_cuda) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, _log_p, pi, mask = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Get log_p corresponding to selected actions
    log_p = _log_p.gather(2, pi.unsqueeze(-1)).squeeze(-1)

    # Optional: mask out actions irrelevant to objective so they do not get reinforced
    if mask is not None:
        log_p[mask] = 0

    assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

    # Calculate loss
    log_likelihood = log_p.sum(1)
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    adaptief_vorige_score, adaptief_current_graph_size, adaptief_current_entropy = loadAdaptief(opts)
    if (len(adaptief_vorige_score) >= (opts.epoch_size/opts.batch_size)*2):
        adaptief_vorige_score.pop(0)
    adaptief_vorige_score.append(float(cost.mean()))
    saveAdaptief(adaptief_vorige_score, adaptief_current_graph_size, adaptief_current_entropy, opts)

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, opts)
        

def loadAdaptief(opts):

    data = {}
    with open("temp/"+opts.run_name+'_adapatief.json') as f:
        data = json.load(f)
    return data["a"], data["b"], data["c"]

def saveAdaptief(a,b,c,opts):



    objectI = {"a":a,"b":b,"c":c}

    with open("temp/"+opts.run_name+'_adapatief.json', 'w') as outfile:
        json.dump(objectI, outfile)


def loadArgument(filename):

    data = {}
    with open("experiments/{}".format(filename)) as f:
        data = json.load(f)
    return data