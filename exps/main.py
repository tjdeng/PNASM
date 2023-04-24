import argparse
import random
import time

import torch
import numpy as np

from exps.controllers.controller_201 import Controller201
from exps.controllers.controller_asr import ControllerASR
from exps.controllers.controller_trainer import ControllerTrainer
from exps.nasbenchs.nasbench201.nasbench201 import Nasbench201
from exps.nasbenchs.nasbenchasr.nasbenchasr import NasbenchASR
from exps.predictors.ensemble import Ensemble
from exps.predictors.train_predictor import train_predictor
from exps.utils import get_config_from_yaml, record_best_arch_info, record_avg_info, get_base_path, get_logger, \
    log_config


def build_args():
    parser = argparse.ArgumentParser(description='RPNASM')
    register_default_args(parser)
    args = parser.parse_args()

    return args


def register_default_args(parser):
    parser.add_argument(
        "--config_file",
        type=str,
        default="",
    )


def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)


def run_search(config, controller_trainer, log):
    start_time = time.time()

    # 100 steps per epoch.
    if config['predictor_mode'] == "adaptive":
        controller_trainer.search_with_predictor_adaptive()
    else:
        controller_trainer.search()

    if config['predictor_mode'] in ["fixed_k", "adaptive"]:
        log.info(20 * "-" + "Retrain the predictor" + 20 * "-")
        train_predictor(config, controller_trainer)
        log.info(20 * '-' + "Finishing training!" + 20 * '-')
        controller_trainer.buffer_unique_archs["pred_info"] = []

    controller_trainer.cur_total_time_costs += time.time() - start_time


def main(config, path=None):
    """
    support two types of search budgets: time_budget and search_steps.
    for nasbench201, using time_budget.
    for nasbenchasr, using search_steps.
    """

    # set a random seed
    torch.set_num_threads(config['workers'])
    prepare_seed(config['rand_seed'])

    # log path
    log_path = path + '/' + str(config['rand_seed'])
    log = get_logger(log_path)

    # record the config
    log_config(log, config)

    # load models
    nasbench = None
    if config['bench_name'] == 'nasbench201':
        nasbench = Nasbench201(config['bench_name'], config['search_space_type'], config['dataset'], config['data_path'])
        controller = Controller201(nasbench)
    elif config['bench_name'] == 'nasbenchasr':
        nasbench = NasbenchASR(config['bench_name'], config['search_space_type'], config['dataset'], config['data_path'])
        controller = ControllerASR(nasbench)
    else:
        raise Exception('Invalid bench_name: {}'.format(config['bench_name']))

    # build the controller's optimizer
    optimizer = torch.optim.Adam(
        controller.parameters(),
        lr=config['controller_lr'],
        betas=(0.5, 0.999),
        weight_decay=config['controller_weight_decay'],
        eps=config['controller_eps'],
    )
    if torch.cuda.is_available():
        controller = controller.cuda()

    # build a predictor
    predictor = None
    if config['is_predictor']:
        print(20 * "-" + "build a predictor." + 20 * "-")
        if config['is_ensemble']:
            predictor = Ensemble(
                num_ensemble=config['num_ensemble'],
                ss_type=config['bench_name'],
                predictor_type=config['predictor_type'],
                dataset=config['dataset'],
                bench_api=nasbench,
            )
        else:
            raise Exception("Invalid config: is_ensemble.")

    # build a controller trainer
    controller_trainer = ControllerTrainer(controller, optimizer, predictor, nasbench, config, log)

    epoch = 0
    # initialize the predictor
    if config['is_predictor']:
        log.info(20 * "-" + "Initializing predictor" + 20 * "-")
        temp = config['predictor_mode']
        config['predictor_mode'] = "None"
        config['is_predictor'] = False
        controller_trainer.search()
        config['predictor_mode'] = temp
        config['is_predictor'] = True
        epoch += 1

        # we refer to the predictor seminas.
        train_predictor(config, controller_trainer)
        log.info(20 * "-" + "Finish Initializing predictor" + 20 * "-")

    # search
    if config['search_type'] == 'time_budget':
        while controller_trainer.cur_total_time_costs < config['search_time']:
            run_search(config, controller_trainer, log)

    elif config['search_type'] == 'search_steps':
        if config['bench_name'] == 'nasbench201':
            while epoch < config['controller_epochs']:
                run_search(config, controller_trainer, log)
                epoch += 1

        elif config['bench_name'] == 'nasbenchasr':
            while len(controller_trainer.buffer_unique_archs["true_info"]) < config['target_steps']:
                run_search(config, controller_trainer, log)

        else:
            raise Exception('Invalid bench_name: {}'.format(config['bench_name']))

    else:
        raise Exception("Invalid search_type:", config['search_type'])

    # record best arch's info
    best_arch_info = controller_trainer.get_cur_best_arch()
    best_arch_info['search_time'] = controller_trainer.cur_total_time_costs
    return best_arch_info


if __name__ == '__main__':
    args = build_args()
    config = get_config_from_yaml(args.config_file)
    print("config:", config)

    base_path = get_base_path(config)
    results_path = base_path + '/' + "results.txt"

    if config['rand_seed'] == -1:
        vals, tests, search_times = [], [], []
        for i in range(config['loops_if_rand']):
            config['rand_seed'] = random.randint(1, 100000)

            results = main(config, base_path)

            # record the best arch's info.
            record_best_arch_info(results, results_path, i, config['rand_seed'])
            vals.append(results['val_acc'])
            tests.append(results['test_acc'])
            search_times.append(results['search_time'])

        record_avg_info(results_path, vals, tests, search_times)
    else:
        results = main(config, base_path)
        record_best_arch_info(results, results_path, 0, config['rand_seed'])


