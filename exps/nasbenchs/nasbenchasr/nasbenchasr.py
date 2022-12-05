import random
import numpy as np

from exps.nasbenchs.nasbench.nasbench import Nasbench
from exps.nasbenchs.nasbenchasr.conversions import flatten, copy_structure
from exps.nasbenchs.nasbenchasr.encodings_asr import encode_adjacency_one_hot, encode_compact, \
    encode_seminas_nasbenchasr
from exps.nasbenchs.nasbenchasr.utils import from_folder

MAIN_OP_NAMES = ['linear', 'conv5', 'conv5d2', 'conv7', 'conv7d2', 'zero']
SKIP_OP_NAMES = ['zero', 'identity']


class NasbenchASR(Nasbench):

    def __init__(self, bench_name, search_space_type, dataset, data_path):
        super().__init__(bench_name, search_space_type, dataset, data_path)
        self.bench_name = bench_name
        self.search_space_type = search_space_type
        self.dataset = dataset
        self.api = from_folder(data_path, include_static_info=True)
        self.edge2index = {'1m0': 0, '1s0': 1, '2m1': 2, '2s0': 3, '2s1': 4, '3m2': 5, '3s0': 6, '3s1': 7, '3s2': 8}
        self.op_names = {"main_op_names": MAIN_OP_NAMES, "skip_op_names": SKIP_OP_NAMES}
        self.max_nodes = 3

    def random_sample_arch(self):
        super().random_sample_arch()
        search_space = [[len(MAIN_OP_NAMES)] + [2] * (idx + 1) for idx in
                        range(self.max_nodes)]
        flat = flatten(search_space)
        m = [random.randrange(opts) for opts in flat]
        arch = copy_structure(m, search_space)

        return str(arch)

    def encode(self, arch: str, encoding_type: str = 'adjacency_one_hot'):
        super().encode(arch, encoding_type)
        if isinstance(arch, str):
            arch = eval(arch)

        if encoding_type == 'adjacency_one_hot':
            return encode_adjacency_one_hot(arch)

        elif encoding_type == 'arch':
            return encode_compact(arch)

        elif encoding_type == 'seminas':
            return encode_seminas_nasbenchasr(arch)

        else:
            print('{} is not yet implemented as an encoding type \
             for asr'.format(encoding_type))
            raise NotImplementedError()

    def get_arch_info(self, arch: str, epoch: int, deterministic: bool = False):
        super().get_arch_info(arch, epoch, deterministic)
        if isinstance(arch, str):
            arch = eval(arch)
        metric = {}
        seeds = [1234]
        val_pers, test_pers = [], []
        for s in seeds:
            query_results = self.api.full_info(arch, seed=s)
            val_pers.append(float(query_results["val_per"][epoch]))
            test_pers.append(query_results["test_per"])

        val_per = np.mean(val_pers)
        test_per = np.mean(test_pers)
        time_cost = -1

        metric['arch'] = arch
        metric['val_acc'] = val_per
        metric['test_acc'] = test_per
        metric['time_cost'] = time_cost
        return metric

    def convert_structure(self, sample: list):
        super().convert_structure(sample)
        nodes_input = [2, 2 + 3, 2 + 3 + 4]
        arch = []
        p1 = 0
        for p2 in nodes_input:
            arch.append(sample[p1:p2])
            p1 = p2

        return str(arch)






