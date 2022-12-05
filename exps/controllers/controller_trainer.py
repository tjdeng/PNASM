import torch
import time
import numpy as np


class ControllerTrainer(object):

    def __init__(self, controller, optimizer, predictor, nasbench, config, log):
        self.controller = controller
        self.optimizer = optimizer
        self.nasbench = nasbench
        self.predictor = predictor
        self.log = log
        self.config = config

        self.buffer_unique_archs = {"true_info": [], "pred_info": []}
        self.buffer_all_archs = []
        self.agent_buffer = []

        self.cur_total_time_costs = 0.0
        self.alter_flag = True

        self.topn = 5
        self.next_group = []

        self.acq_fn = None

    def set_predictor(self, predictor):
        self.predictor = predictor

    def set_acq_fn(self, acq_fn):
        self.acq_fn = acq_fn

    def get_unique_random_arch(self):
        sampled_arch = self.nasbench.random_sample_arch()

        while sampled_arch in self.buffer_all_archs:
            sampled_arch = self.nasbench.random_sample_arch()

        return sampled_arch

    def get_unique_arch(self):
        probs, log_prob, entropy, sampled_arch, actions = self.controller()

        while sampled_arch in self.buffer_all_archs:
            probs, log_prob, entropy, sampled_arch, actions = self.controller()

        return probs, log_prob, entropy, sampled_arch, actions

    def record_diff_true_pred_val(self, arch, pred_val):
        metric = self.nasbench.get_arch_info(arch, epoch=12)
        val_top1 = metric['val_acc']
        diff = abs(val_top1 - pred_val)
        self.log.info("Arch: {0}, true_val: {1}, pred_val: {2}, diff: {3}".format(arch, val_top1, pred_val, diff))

    def controller_sample(self, steps, baseline, is_predictor):

        decay = 0.95
        controller_entropy_weight = 0.0001

        # global current_total_costs
        for step in range(steps):
            if self.cur_total_time_costs > self.config['search_time'] or len(self.buffer_unique_archs["true_info"]) > self.config['target_steps']:
                break

            probs, log_prob, entropy, sampled_arch, actions = self.get_unique_arch()

            if is_predictor:
                time_start = time.time()
                if not self.config['is_ensemble']:
                    val_top1 = self.acq_fn.query([sampled_arch])
                else:
                    val_top1 = self.acq_fn(sampled_arch)
                time_cost = time.time() - time_start
                self.buffer_unique_archs["pred_info"].append([sampled_arch, val_top1])

            else:
                if self.config['bench_name'] == 'nasbench201':
                    metric = self.nasbench.get_arch_info(sampled_arch, epoch=12)
                    val_top1 = metric['val_acc']
                elif self.config['bench_name'] == 'nasbenchasr':
                    metric = self.nasbench.get_arch_info(sampled_arch, epoch=-1)
                    val_top1 = -metric['val_acc']
                else:
                    raise Exception("not support benchmark:", self.config['bench_name'])

                time_cost = metric['time_cost']
                self.buffer_unique_archs["true_info"].append([sampled_arch, val_top1])

            # self.record_diff_true_pred_val(sampled_arch, val_top1)
            self.buffer_all_archs.append(sampled_arch)

            val_top1 = torch.tensor(val_top1)
            reward = val_top1 + controller_entropy_weight * entropy

            if baseline is None:
                baseline = val_top1
            else:
                # baseline = prev_baseline - (1 - controller_bl_dec) * (
                #     prev_baseline - reward
                # )
                baseline = decay * baseline + (1 - decay) * reward

            self.log.info(
                "Is_predictor: {:}, Baseline: {:.3f}, Val_acc: {:.4f}, Reward:{:.4f}, Time_cost: {:4f}, Arch: {:}".format(
                    is_predictor, baseline.item(), val_top1, reward.item(), time_cost, sampled_arch
                )
            )

            if torch.is_tensor(reward):
                reward = reward.tolist()
            if torch.is_tensor(log_prob):
                log_prob = log_prob.tolist()

            self.agent_buffer.append((actions, log_prob, reward))
            self.cur_total_time_costs += time_cost

        return baseline

    def cal_loss(self, log_p, log_old_p, reward, baseline):
        if self.config['update_controller_algo'] == "ppo":
            ratio = torch.exp(log_p - log_old_p)
            adv = reward - baseline
            clip_adv = torch.clamp(ratio, 1 - self.config['clip_ratio'], 1 + self.config['clip_ratio']) * adv
            policy_loss = -(torch.min(ratio * adv, clip_adv))

        elif self.config['update_controller_algo'] == "pg":
            policy_loss = -1 * log_p * (reward - baseline)

        else:
            raise ValueError("Invalid agent's algo: {:}".format(self.config['update_controller_algo']))

        return policy_loss

    def train_policy(self, baseline):

        for i in range(self.config['update_controller_iters']):
            loss = 0
            for v in self.agent_buffer:
                actions, log_old_prob, reward = v

                actions = torch.as_tensor(actions)
                log_old_prob = torch.as_tensor(log_old_prob)
                reward = torch.as_tensor(reward)

                if torch.cuda.is_available():
                    actions = actions.cuda()
                    log_old_prob = log_old_prob.cuda()
                    reward = reward.cuda()

                probs, log_prob, _, _ = self.controller.get_prob(actions)
                # print("log_old_prob:", log_old_prob, " log_prob: ", log_prob)
                loss += self.cal_loss(log_prob, log_old_prob, reward, baseline.detach())

            loss /= len(self.agent_buffer)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print("update_controller_algo: ", self.config['update_controller_algo'], " ",
            #       "update_controller_iters: ", self.config['update_controller_iters'], " ",
            #       "size_agent_buffer: ", len(self.agent_buffer), " ",
            #       "batchs_per_epoch: ", self.config['batchs_per_epoch'], " ",
            #       )

    def get_cur_best_arch(self):
        sorted_true_data = sorted(self.buffer_unique_archs["true_info"], key=lambda v: (v[1], v[0]), reverse=True)
        # choose the top1
        best_arch, best_val = sorted_true_data[0][0], sorted_true_data[0][1]
        # print("best_arch:", best_arch, "sorted_true_data[0]:", sorted_true_data[0])

        pred_best_val = float('-inf')
        best_pred_arch = None
        if len(self.buffer_unique_archs["pred_info"]) > 0:
            if len(self.next_group) == 0:
                sorted_pred_data = sorted(self.buffer_unique_archs["pred_info"], key=lambda v: (v[1], v[0]), reverse=True)
                top_n = sorted_pred_data[:self.topn]
                top_n.reverse()
                self.next_group = [arch_info for arch_info in top_n]

            best_pred_arch = self.next_group.pop()[0]

            # print("dataset, best_pred_arch:", dataset, best_pred_arch,)
            true_info_archs = np.array(self.buffer_unique_archs["true_info"])[:, 0].tolist()
            if best_pred_arch in true_info_archs:
                idx = true_info_archs.index(best_pred_arch)
                pred_best_val = self.buffer_unique_archs["true_info"][idx][1]
            else:
                if self.config['bench_name'] == 'nasbench201':
                    metric = self.nasbench.get_arch_info(best_pred_arch, epoch=12)
                    val_top1 = metric['val_acc']
                elif self.config['bench_name'] == 'nasbenchasr':
                    metric = self.nasbench.get_arch_info(best_pred_arch, epoch=-1)
                    val_top1 = -metric['val_acc']
                else:
                    raise Exception("not support benchmark:", self.config['bench_name'])

                time_cost = metric['time_cost']
                pred_best_val = val_top1
                self.buffer_unique_archs["true_info"].append([best_pred_arch, val_top1])
                self.cur_total_time_costs += time_cost

        if pred_best_val > best_val:
            metric = self.record_cur_best_arch_info(best_pred_arch, "pred_info")
        else:
            metric = self.record_cur_best_arch_info(best_arch, "true_info")

        return metric

    def record_cur_best_arch_info(self, best_arch, from_where):

        if self.config['bench_name'] == 'nasbench201':
            metric = self.nasbench.get_arch_info(best_arch, epoch=-1)
            val_top1 = metric['val_acc']
        elif self.config['bench_name'] == 'nasbenchasr':
            metric = self.nasbench.get_arch_info(best_arch, epoch=-1)
            val_top1 = -metric['val_acc']
        else:
            raise Exception("not support benchmark:", self.config['bench_name'])

        test_top1 = metric['test_acc']
        self.log.info("From: {}, cur_best_arch: {}, cur_total_time_costs: {}, val_acc: {}, test_acc: {}".format(
            from_where,
            best_arch,
            self.cur_total_time_costs,
            val_top1,
            test_top1
        )
        )

        return metric

    def train_controller(self, baseline, t_steps=20):

        self.controller.train()
        self.controller.zero_grad()
        self.agent_buffer = []

        p_steps = 0
        if self.config['is_predictor']:
            p_steps = self.config['episodes'] - t_steps

        baseline = self.controller_sample(t_steps, baseline, False)

        if self.config['is_predictor']:
            baseline = self.controller_sample(p_steps, baseline, True)

        num_sampled_archs = len(self.buffer_all_archs)
        if num_sampled_archs % self.config['steps_each_record'] == 0:
            _ = self.get_cur_best_arch()

        self.train_policy(baseline)

        return baseline

    def search(self, baseline=None):
        for i in range(self.config['batchs_per_epoch']):
            if self.cur_total_time_costs > self.config['search_time'] or len(self.buffer_unique_archs["true_info"]) > self.config['target_steps']:
                break

            if self.config['predictor_mode'] == "None":
                t_steps = self.config['episodes']
            elif self.config['predictor_mode'] == "all":
                t_steps = 0
            elif self.config['predictor_mode'] == "fixed_k":
                t_steps = self.config['fixedk']
            else:
                raise Exception("Invalid predictor mode:", self.config['predictor_mode'])

            baseline = self.train_controller(baseline, t_steps=t_steps)

    def search_with_predictor_adaptive(self, baseline=None):

        # collect one batch data.
        temp_buffer = []
        for i in range(self.config['episodes']):
            probs, log_prob, entropy, sampled_arch, actions = self.get_unique_arch()
            if torch.is_tensor(probs):
                probs = probs.tolist()

            temp_buffer.append((actions, probs))

        i = 0
        while i < self.config['batchs_per_epoch']:
            if self.cur_total_time_costs > self.config['search_time'] or len(self.buffer_unique_archs["true_info"]) > self.config['target_steps']:
                break

            alpha = self.kl_check(temp_buffer)

            if alpha <= 0:
                alpha = 0
            elif alpha >= 1:
                alpha = 1

            adap_steps = int(alpha * self.config['episodes'])

            # alpha = self.kl_check(temp_buffer)
            # adap_steps = int(steps / (1 + math.exp(-alpha)))
            self.log.info("Alpha: {}, steps: {}, true_steps: {}, pred_steps: {}".format(alpha, self.config['episodes'], adap_steps, self.config['episodes'] - adap_steps))
            baseline = self.train_controller(baseline, t_steps=adap_steps)

            i += 1

    def kl_check(self, buffer):
        kl_loss = 0
        for v in buffer:
            actions, target = v

            actions = torch.as_tensor(actions)
            if torch.cuda.is_available():
                actions = actions.cuda()
            #     target = target.cuda()

            preds, _, _, _ = self.controller.get_prob(actions)

            cur_loss = 0
            for i in range(len(target)):
                cur_loss += torch.nn.functional.kl_div(torch.log(torch.tensor(preds[i]) + 1e-5 ), torch.tensor(target[i]), reduction='sum').item()

            kl_loss += cur_loss / len(target)

        avg_kl_loss = kl_loss / len(buffer)
        self.log.info("kl_loss: {}, avg_kl_loss: {}".format(kl_loss, avg_kl_loss))
        return kl_loss
