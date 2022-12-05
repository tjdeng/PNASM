import time

from exps.predictors.acquisition_functions import acquisition_function


def train_predictor(config, controller_trainer):
    start_time = time.time()
    if config['is_ensemble']:

        xtrain = [v[0] for v in controller_trainer.buffer_unique_archs["true_info"]]
        ytrain = [v[1] for v in controller_trainer.buffer_unique_archs["true_info"]]

        if config['predictor_type'] == "seminas":
            unlabeled = []
            # create unlabeled data and pass it to the predictor
            while len(unlabeled) < len(xtrain):
                arch = controller_trainer.get_unique_random_arch()
                unlabeled.append(arch)
            controller_trainer.predictor.set_pre_computations(
                unlabeled=unlabeled
            )

        train_error = controller_trainer.predictor.fit(xtrain, ytrain)
        # print("train_error: ", train_error)
        # define an acquisition function
        acq_fn = acquisition_function(
            ensemble=controller_trainer.predictor, ytrain=None, acq_fn_type="exploit_only"
        )

    controller_trainer.acq_fn = acq_fn
    controller_trainer.cur_total_time_costs += time.time() - start_time

