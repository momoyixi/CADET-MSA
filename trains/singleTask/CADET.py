import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch.nn.functional as F
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss
import os

logger = logging.getLogger('MMSA')


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


class CADET():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.orthogonality_loss = nn.CosineEmbeddingLoss()
        self.contrastive_loss = HingeLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()

    def do_train(self, model, dataloader, return_epoch_results=False):
        params = model[0].parameters()
        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            verbose=True,
            patience=self.args.patience
        )

        epochs, best_epoch = 0, 0

        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }

        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_cadet = model[0]
        net.append(net_cadet)
        model = net
        use_adv = self.args.use_adversarial

        lambda_adv = getattr(self.args, 'lambda_adv', 0.0)
        lambda_recon_geo = getattr(self.args, 'lambda_recon_geo', 0.1)
        lambda_cl_o = getattr(self.args, 'lambda_cl_o', 0.01)
        lambda_dis = getattr(self.args, 'lambda_dis', 1.0)
        lambda_cyc = getattr(self.args, 'lambda_cyc', 0.0)
        base_lambda_gan = getattr(self.args, 'lambda_gan', 0.0)
        lambda_dpcc = getattr(self.args, 'lambda_dpcc', 1.0)
        adv_warmup_epochs = getattr(self.args, 'adv_warmup_epochs', 0)
        gan_decay_epoch = getattr(self.args, 'gan_decay_epoch', 5)
        gan_decay_rate = getattr(self.args, 'gan_decay_rate', 0.9)

        while True:
            epochs += 1
            y_pred, y_true = [], []

            for mod in model:
                mod.train()

            train_loss = 0.0
            left_epochs = self.args.update_epochs

            if epochs > gan_decay_epoch:
                decay_steps = epochs // gan_decay_epoch
                current_lambda_gan = base_lambda_gan * (gan_decay_rate ** decay_steps)
            else:
                current_lambda_gan = base_lambda_gan

            current_lambda_adv = lambda_adv

            if (not use_adv) or (epochs <= adv_warmup_epochs):
                current_lambda_gan = 0.0
                current_lambda_adv = 0.0

            with tqdm(dataloader['train']) as td:
                for batch_data in td:

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()

                    left_epochs -= 1

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    output = model[0](text, audio, vision)

                    loss_fusion = self.criterion(output['output_logit'], labels)

                    loss_specific_l = self.criterion(output['logits_l_hetero'], labels)
                    loss_specific_v = self.criterion(output['logits_v_hetero'], labels)
                    loss_specific_a = self.criterion(output['logits_a_hetero'], labels)
                    loss_shared = self.criterion(output['logits_c'], labels)

                    loss_msa = (
                        loss_fusion
                        + loss_shared
                        + 3 * loss_specific_l
                        + loss_specific_v
                        + loss_specific_a
                    )

                    loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                    loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                    loss_recon_a = self.MSE(output['recon_a'], output['origin_a'])
                    loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    loss_geo_l = self.MSE(
                        output['s_l'].permute(1, 2, 0),
                        output['s_l_r']
                    )
                    loss_geo_v = self.MSE(
                        output['s_v'].permute(1, 2, 0),
                        output['s_v_r']
                    )
                    loss_geo_a = self.MSE(
                        output['s_a'].permute(1, 2, 0),
                        output['s_a_r']
                    )
                    loss_geo = loss_geo_l + loss_geo_v + loss_geo_a


                    if self.args.dataset_name == 'mosi':
                        num = 50
                    elif self.args.dataset_name == 'mosei':
                        num = 10
                    elif self.args.dataset_name == 'sims':
                        num = 50
                    else:
                        num = output['s_l'].shape[-1]

                    target_o = torch.tensor([-1]).to(self.args.device)

                    loss_o_l = self.orthogonality_loss(
                        output['s_l'].reshape(-1, num),
                        output['c_l'].reshape(-1, num),
                        target_o
                    )
                    loss_o_v = self.orthogonality_loss(
                        output['s_v'].reshape(-1, num),
                        output['c_v'].reshape(-1, num),
                        target_o
                    )
                    loss_o_a = self.orthogonality_loss(
                        output['s_a'].reshape(-1, num),
                        output['c_a'].reshape(-1, num),
                        target_o
                    )

                    loss_o = loss_o_l + loss_o_v + loss_o_a

                    c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']

                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        feats.append(c_l[i].view(1, -1))
                        feats.append(c_v[i].view(1, -1))
                        feats.append(c_a[i].view(1, -1))

                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))

                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)

                    loss_cl = self.contrastive_loss(ids, feats)


                    if hasattr(model[0], 'domain_label_l'):

                        loss_dom = 0.0

                        if output['domain_label_l'] is not None:
                            loss_dom += F.binary_cross_entropy(
                                output['domain_label_l'],
                                torch.ones_like(output['domain_label_l'])
                            )
                            loss_dom += F.binary_cross_entropy(
                                output['domain_label_v'],
                                torch.ones_like(output['domain_label_v'])
                            )
                            loss_dom += F.binary_cross_entropy(
                                output['domain_label_a'],
                                torch.ones_like(output['domain_label_a'])
                            )
                            loss_dom /= 3.0

                        loss_sp = 0.0
                        loss_sp += F.binary_cross_entropy(
                            output['shared_or_private_l'],
                            torch.ones_like(output['shared_or_private_l'])
                        )
                        loss_sp += F.binary_cross_entropy(
                            output['shared_or_private_v'],
                            torch.ones_like(output['shared_or_private_v'])
                        )
                        loss_sp += F.binary_cross_entropy(
                            output['shared_or_private_a'],
                            torch.ones_like(output['shared_or_private_a'])
                        )
                        loss_sp += F.binary_cross_entropy(
                            output['shared_or_private_c'],
                            torch.zeros_like(output['shared_or_private_c'])
                        )
                        loss_sp /= 4.0

                        loss_adv = loss_dom + loss_sp

                    else:
                        loss_adv = 0.0


                    loss_dis = (
                        (loss_recon + loss_geo) * lambda_recon_geo
                        + (loss_cl + loss_o) * lambda_cl_o
                        + loss_adv * current_lambda_adv
                    )


                    loss_cycle_a = self.MSE(output['recon_L_prime'], output['origin_l'])
                    loss_cycle_v = self.MSE(output['recon_L_prime_v'], output['origin_l'])
                    loss_cycle = loss_cycle_a + loss_cycle_v

                    loss_gan_g_a = F.binary_cross_entropy(
                        output['d_fake_A'],
                        torch.ones_like(output['d_fake_A'])
                    )
                    loss_gan_d_a = (
                        F.binary_cross_entropy(
                            output['d_real_A'],
                            torch.ones_like(output['d_real_A'])
                        )
                        + F.binary_cross_entropy(
                            output['d_fake_A'],
                            torch.zeros_like(output['d_fake_A'])
                        )
                    )
                    loss_gan_a = loss_gan_g_a + loss_gan_d_a

                    loss_gan_g_v = F.binary_cross_entropy(
                        output['d_fake_V'],
                        torch.ones_like(output['d_fake_V'])
                    )
                    loss_gan_d_v = (
                        F.binary_cross_entropy(
                            output['d_real_V'],
                            torch.ones_like(output['d_real_V'])
                        )
                        + F.binary_cross_entropy(
                            output['d_fake_V'],
                            torch.zeros_like(output['d_fake_V'])
                        )
                    )
                    loss_gan_v = loss_gan_g_v + loss_gan_d_v

                    loss_gan = loss_gan_a + loss_gan_v

                    loss_dpcc = (
                        loss_cycle * lambda_cyc
                        + loss_gan * current_lambda_gan
                    )

                    loss_total = (
                        loss_msa
                        + lambda_dis * loss_dis
                        + lambda_dpcc * loss_dpcc
                    )

                    loss_total.backward()

                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters())
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += loss_total.item()

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs

                if not left_epochs:
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)

            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN -({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )

            # validation
            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            test_results = self.do_test(model[0], dataloader['test'], mode="TEST")

            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])

            # save each epoch model
            # torch.save(model[0].state_dict(), './pt/' + str(self.args.dataset_name) + '_' + str(epochs) + '.pth')

            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)

            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                model_save_path = './pt/CADET_' + str(self.args.dataset_name) + '.pth'
                # torch.save(model[0].state_dict(), model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)

                test_results = self.do_test(model[0], dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)

            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):

        model.eval()
        y_pred, y_true = [], []

        eval_loss = 0.0

        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }

        feats_l, feats_v, feats_a, feats_c = [], [], [], []
        all_labels = []
        last_hs_list = []

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    output = model(text, audio, vision)

                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

                    last_hs_list.append(output['last_hs_proj'].cpu())
                    all_labels.append(labels.cpu())

        if len(last_hs_list) > 0:
            last_hs_proj = torch.cat(last_hs_list, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            os.makedirs("/root/CADET/pro", exist_ok=True)
            torch.save(
                {"features": last_hs_proj, "labels": all_labels},
                f"/root/CADET/pro/{mode}_last_hs_proj.pt"
            )

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)

        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results

            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)

            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results