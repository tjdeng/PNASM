import random
import numpy as np

from exps.nasbenchs.nasbench201.genotypes import Structure
from exps.nasbenchs.nasbench.nasbench import Nasbench
from nats_bench import create

INPUT = 'input'
OUTPUT = 'output'
OPS = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']
NUM_OPS = len(OPS)
OP_SPOTS = 6
LONGEST_PATH_LENGTH = 3


class Nasbench201(Nasbench):

    def __init__(self, bench_name, search_space_type, dataset, data_path):
        super().__init__(bench_name, search_space_type, dataset, data_path)
        self.bench_name = bench_name
        self.search_space_type = search_space_type
        self.dataset = dataset
        self.api = create(None, search_space_type, fast_mode=True, verbose=False)
        self.edge2index = {'1<-0': 0, '2<-0': 1, '2<-1': 2, '3<-0': 3, '3<-1': 4, '3<-2': 5}
        self.op_names = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        self.max_nodes = 4

    def random_sample_arch(self):
        super().random_sample_arch()
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                op_name = random.choice(self.op_names)
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes).tostr()

    def encode(self, arch: str, encoding_type: str = 'gcn'):
        super().encode(arch, encoding_type)
        op_map = [OUTPUT, INPUT, *OPS]
        ops = self.get_op_list(arch)
        ops_idx = [OPS.index(v) for v in ops]
        ops = [INPUT, *ops, OUTPUT]
        ops_onehot = np.array([[i == op_map.index(op) for i in range(len(op_map))] for op in ops], dtype=np.float32)

        ops = [op + 1 for op in ops_idx]
        ops = [0, *ops, 6]

        matrix = np.array(
            [[0, 1, 1, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0]])

        metric = self.get_arch_info(arch, epoch=12)
        encode_info = {
            'num_vertices': 8,
            'adjacency': matrix,
            'operations_oneshot': ops_onehot,
            'operations': ops,
            'mask': np.array([i < 8 for i in range(8)], dtype=np.float32),
            'val_acc':  metric['val_acc'],
            'test_acc': metric['test_acc'],
            'time_cost': metric['time_cost'],
        }

        return encode_info

    def get_arch_info(self, arch: str, epoch: int, deterministic: bool = False):
        super().get_arch_info(arch, epoch, deterministic)
        metric = {}
        if epoch == 12:
            test_acc = 0.0
            # val_acc, _, time_cost, _ = self.api.simulate_train_eval(
            #     arch, self.dataset, hp="12"
            # )

            if self.dataset == "cifar10":
                info = self.api.get_more_info(arch, "cifar10-valid", hp="12", is_random=deterministic)
            else:
                info = self.api.get_more_info(arch, self.dataset, hp="12", is_random=deterministic)

            val_acc, time_cost = info["valid-accuracy"], info["train-all-time"] + info["valid-per-time"]

        elif epoch == -1:
            if self.dataset == "cifar10":
                xinfo = self.api.get_more_info(
                    arch,
                    dataset=self.dataset,
                    hp="200" if self.search_space_type == "tss" else "90",
                    is_random=deterministic
                )
                test_acc = xinfo["test-accuracy"]
                xinfo = self.api.get_more_info(
                    arch,
                    dataset="cifar10-valid",
                    hp="200" if self.search_space_type == "tss" else "90",
                    is_random=deterministic,
                )
                val_acc = xinfo["valid-accuracy"]
            else:
                xinfo = self.api.get_more_info(
                    arch,
                    dataset=self.dataset,
                    hp="200" if self.search_space_type == "tss" else "90",
                    is_random=deterministic
                )
                val_acc = xinfo["valid-accuracy"]
                test_acc = xinfo["test-accuracy"]

            time_cost = xinfo["train-all-time"] + xinfo["valid-per-time"]

        else:
            raise Exception("Invalid epoch {0} for the api {1}.".format(epoch, self.bench_name))

        metric['arch'] = arch
        metric['val_acc'] = val_acc
        metric['test_acc'] = test_acc
        metric['time_cost'] = time_cost
        return metric

    def convert_structure(self, sample: list):
        super().convert_structure(sample)
        genotypes = []
        for i in range(1, self.max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                op_index = sample[self.edge2index[node_str]]
                op_name = self.op_names[op_index]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return Structure(genotypes).tostr()

    def get_op_list(self, string: str):
        # given a string, get the list of operations
        tokens = string.split('|')
        ops = [t.split('~')[0] for i, t in enumerate(tokens) if i not in [0, 2, 5, 9]]

        return ops







