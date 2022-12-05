import logging
import sys
import os
import time

import yaml
import numpy as np


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self, file_name, level='info',
                 fmt='%(asctime)s - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(file_name)
        self.file_name = file_name

        # 设置日志格式
        self.format_str = logging.Formatter(fmt)

        # 设置日志级别
        self.logger.setLevel(self.level_relations.get(level))

    def output_console(self):
        # 往屏幕上输出
        sh = logging.StreamHandler()
        sh.setFormatter(self.format_str)
        self.logger.addHandler(sh)
        return sh

    def output_file(self):
        # 将日志写到文件里面
        th = logging.FileHandler(self.file_name,
                                 encoding='utf-8')
        th.setFormatter(self.format_str)
        self.logger.addHandler(th)
        return th

    def remove_handler(self, handler):
        self.logger.removeHandler(handler)

    def __call__(self, message):
        sh = self.output_console()
        th = self.output_file()

        self.logger.info(message)

        self.remove_handler(sh)
        self.remove_handler(th)


def get_logger(log_path):
    time_format = "%m/%d %H:%M:%S"
    fmt = "[%(asctime)s] %(levelname)s (%(name)s) %(message)s"
    formatter = logging.Formatter(fmt, time_format)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    fileHandler = logging.FileHandler(filename=log_path + "_" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    fileHandler.setLevel(logging.INFO)
    formatterFile = logging.Formatter("%(message)s")
    fileHandler.setFormatter(formatterFile)
    logger.addHandler(fileHandler)

    return logger


def load_info(path):
    with open(path, 'r', encoding='utf-8') as f:
        info = yaml.load(f.read(), Loader=yaml.FullLoader)

    return info


def get_config_from_yaml(path):
    if path == "":
        raise Exception("Invalid path.")

    # load config from path.
    config = load_info(path)

    if isinstance(config['search_time'], str):
        config['search_time'] = eval(config['search_time'])
    if isinstance(config['target_steps'], str):
        config['target_steps'] = eval(config['target_steps'])

    return config


def record_best_arch_info(results, path, idx, seed):
    with open(path, 'a') as f:
        f.write(
            "Seed: {0}, Num: {1}, Arch: {2}, Val: {3}, Test: {4}, Total search time: {5}\n".format(
                seed,
                idx,
                results['arch'],
                results['val_acc'],
                results['test_acc'],
                results['search_time']
            )
        )


def record_avg_info(path, vals, tests, search_times):
    with open(path, 'a') as f:
        f.write("Avg vals: {0}, std: {1}".format(np.mean(vals), np.std(vals)))
        f.write("\n")

        f.write("Avg tests: {0}, std: {1}".format(np.mean(tests), np.std(tests)))
        f.write("\n")

        f.write("Avg search times: {0}, std: {1}".format(np.mean(search_times), np.std(search_times)))
        f.write("\n")


def get_base_path(config):
    path = config['log_path'] + '/' + \
                config['bench_name'] + '/' + \
                config['dataset']

    if config['is_predictor']:
        base_path = path + '/' + \
                   "predictor" + '/' + \
                    config['predictor_type'] + '_' + \
                    config['predictor_mode'] + str(config['fixedk']) + '_'

    else:
        base_path = path + '/' + \
                    "wo_predictor" + '/'

    base_path = base_path + \
                config['update_controller_algo'] + '_' + \
                "batch" + str(config['episodes'])

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    return base_path


def log_config(log, config):
    keys = list(config.keys())
    for k in keys:
        log.info("{0}: {1}".format(k, config[k]))







