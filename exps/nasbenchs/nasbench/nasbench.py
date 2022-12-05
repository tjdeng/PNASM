

class Nasbench(object):
    """
    support a benchmark that builds one arch with the way of static DAG.
    """

    def __init__(self, bench_name: str, search_space_type: str, dataset: str, data_path: str):
        self.bench_name = bench_name
        self.search_space_type = search_space_type
        self.dataset = dataset
        self.api = None

    def random_sample_arch(self):
        """
        randomly sample a arch according to candidate operations.
        :return: arch: a string
        """
        pass

    def encode(self, arch: str, encoding_type: str):
        """
        encode a arch according to the encoding_type.
        :param arch: str
        :param encoding_type:
        :return: encoding_info, such as the encoding info of nasbench201:
            'num_vertices': 8,
            'adjacency': matrix,
            'operations_oneshot': ops_onehot,
            'operations': ops,
            'mask': np.array([i < 8 for i in range(8)], dtype=np.float32),
            'val_acc': val_acc,
            'test_acc': test_acc,
            'time_cost': time_cost,
        """
        pass

    def get_arch_info(self, arch: str, epoch: int, deterministic: bool):
        """
        get arch's metric by the api.
        :param arch:
        :param epoch:
        :param deterministic:
        :return: metric: a dict
        """
        pass

    def convert_structure(self, sample: list):
        """
        convert a list sampled by a controller into the form of arch.
        :param sample:
        :return: arch: a string
        """
        pass



