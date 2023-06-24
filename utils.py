import os
import dill
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import data_loader
import pickle
from make_models import *
from random import randrange
import numpy as np
from randaugment import RandAugment_face as RandAugment
import csv
from metrics import *


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class AverageMeterVector(object):
    """Computes and stores the average and current value"""

    def __init__(self, N):
        self.reset(N)

    def reset(self, N):
        self.avg = np.zeros(N)
        self.sum = np.zeros(N)
        self.count = np.zeros(N)

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,), exact=False):
    with torch.no_grad():
        # Binary Classification
        if len(target.shape) > 1:
            assert output.shape == target.shape, \
                "Detected binary classification but output shape != target shape"
            return [torch.round(torch.sigmoid(output)).eq(torch.round(target)).float().mean()], [
                -1.0]

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_exact = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float()
            ck_sum = correct_k.sum(0, keepdim=True)
            res.append(ck_sum.mul_(100.0 / batch_size))
            res_exact.append(correct_k)

        if not exact:
            return res
        else:
            return res_exact


def group_accuracy(args, output, target, group):
    with torch.no_grad():
        batch_size = target.size(0)
        exact_acc = accuracy(output, target, topk=(1,), exact=True)[0]
        group_correct = torch.zeros((args.num_groups, args.num_labels))
        group_cnt = torch.zeros((args.num_groups, args.num_labels))
        for g in range(args.num_groups):
            for y in range(args.num_labels):
                group_exact_acc = exact_acc[(group == g) * (target == y)]
                group_correct[g, y] = group_exact_acc.sum()
                group_cnt[g, y] = len(group_exact_acc)
        if batch_size != group_cnt.sum().item():
            err_msg = "Errors in computing group accuracy!"
            raise ValueError(err_msg)

    return group_correct, group_cnt


def eql_op(group_acc, y=1):
    with torch.no_grad():
        group_acc_y = group_acc[:, y]
        min_acc, min_idx = torch.min(group_acc_y, dim=0)
        max_acc, max_idx = torch.max(group_acc_y, dim=0)
        diff = max_acc - min_acc

    return diff, max_idx, min_idx


def eql_odd(group_acc):
    with torch.no_grad():
        group_num, label_num = group_acc.shape
        diff_matrix = torch.zeros((group_num, group_num))
        for i in range(group_num):
            for j in range(i + 1, group_num):
                for y in range(label_num):
                    diff_matrix[i, j] += abs(group_acc[i, y] - group_acc[j, y])
                    # diff_matrix[j, i] = diff_matrix[i, j]
        diff = torch.max(diff_matrix)
        i, j = torch.nonzero(diff_matrix == diff, as_tuple=True)
        i = i[0].item()
        j = j[0].item()
        if group_acc[i, 1] > group_acc[j, 1]:
            return diff, i, j
        else:
            return diff, j, i


def acc_disparity(group_correct, group_cnt):
    with torch.no_grad():
        group_correct = group_correct.sum(axis=1)
        group_cnt = group_cnt.sum(axis=1)
        group_acc = torch.nan_to_num(group_correct / group_cnt)
        min_acc, min_idx = torch.min(group_acc, dim=0)
        max_acc, max_idx = torch.max(group_acc, dim=0)
        diff = max_acc - min_acc

    return diff, max_idx, min_idx


def accuracy_cifar(output, target, topk=(1,), per_class=False):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    if per_class:
        # per class accuracy, only top1
        num_classes = output.size(1)

        res_per_class = torch.zeros(num_classes)
        rec_num = torch.zeros(num_classes)
        for class_i in range(num_classes):
            correct_class = correct * (target.view(1, -1) == class_i).expand_as(pred)
            correct_k = correct_class[0].reshape(-1).float().sum(0)
            rec_num[class_i] = torch.sum(target == class_i)
            res_per_class[class_i] = (correct_k.mul_(100.0 / rec_num[class_i])) if rec_num[
                                                                                       class_i] > 0 else 0.0
        return res_per_class, rec_num
    else:
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def statistic(dataset, args):
    groups = {'White_Male': 0, 'Black_Male': 0, 'White_Female': 0, 'Black_Female': 0}

    for i, sample in enumerate(dataset):
        gender = sample['label']['gender']
        race = sample['label']['race']
        if gender == 0 and race == 0:
            groups['White_Male'] += 1
        elif gender == 0 and race == 1:
            groups['Black_Male'] += 1
        elif gender == 1 and race == 0:
            groups['White_Female'] += 1
        elif gender == 1 and race == 1:
            groups['Black_Female'] += 1

    return groups


def save_checkpoint(args, filename, sd_info):
    ckpt_save_path = os.path.join(args.save_path, args.save_name)
    if not os.path.exists(ckpt_save_path):
        os.makedirs(ckpt_save_path)
    torch.save(sd_info, os.path.join(ckpt_save_path, filename), pickle_module=dill)


def eval_loop(args, epoch, domain_type, dataloader, model):
    domain_msg = 'Source' if domain_type == 'source' else 'Target'

    # init statistics
    losses = AverageMeter()
    accs = AverageMeter()
    group_correct = torch.zeros((args.num_groups, args.num_labels))
    group_cnt = torch.zeros((args.num_groups, args.num_labels))

    # switch to eval mode
    model.eval()

    # training criterion
    loss_fn = nn.CrossEntropyLoss()

    # dataloader
    iterator = enumerate(dataloader)

    with torch.no_grad():
        for i, sample_batch in iterator:
            if args.dataset == 'utk-fairface':
                inputs = sample_batch['image']
                inputs = inputs.to(args.device)
                labels = sample_batch['label']['gender'].to(args.device)
                groups = sample_batch['label']['race'].to(args.device)
            elif args.dataset == 'shapes':
                inputs = sample_batch[0].float().to(args.device)
                labels = sample_batch[1].long().squeeze().to(args.device)
                groups = sample_batch[2].long().squeeze().to(args.device)
            elif args.dataset == 'newadult':
                inputs, _ = sample_batch[0]
                inputs = inputs.float().to(args.device)
                labels = sample_batch[1].long().squeeze().to(args.device)
                groups = sample_batch[2].long().squeeze().to(args.device)
            else:
                raise Exception(f'Unknown dataset: {args.dataset}')

            # forward
            outputs, features = model(inputs)

            loss = loss_fn(outputs, labels)
            losses.update(loss.item(), inputs.size(0))

            # statistics
            prec = accuracy(outputs, labels)[0]
            accs.update(prec.item(), inputs.size(0))
            batch_group_correct, batch_group_cnt = group_accuracy(args, outputs, labels, groups)
            group_correct += batch_group_correct
            group_cnt += batch_group_cnt

        # measure unfairness
        acc_dis, max_id_acc, min_id_acc = acc_disparity(group_correct, group_cnt)
        group_acc = torch.nan_to_num(group_correct / group_cnt) * 100
        acc_var = torch.std(group_acc, unbiased=False)
        err_op0, max_id_op0, min_id_op0 = eql_op(group_acc, 0)
        err_op1, max_id_op1, min_id_op1 = eql_op(group_acc)
        err_odd, max_id_odd, min_id_odd = eql_odd(group_acc)

        # log
        print('Val {0} Epoch:{1} | Loss {loss:.4f} | Acc {acc:.2f} '
              '[{acc_a0_y0:.2f} {acc_a0_y1:.2f} {acc_a1_y0:.2f} {acc_a1_y1:.2f}]|'
              'acc_var {acc_var:.2f}|'
              'acc_dis {acc_dis:.2f}, ({max_id_acc}, {min_id_acc})|'
              'err_op_0 {err_op0:.2f}, ({max_id_op0}, {min_id_op0})|'
              'err_op_1 {err_op1:.2f}, ({max_id_op1}, {min_id_op1})|'
              'err_odd {err_odd:.2f}, ({max_id_odd}, {min_id_odd})|'.format(
            domain_msg, epoch, loss=losses.avg, acc=accs.avg, acc_var=acc_var,
            err_op0=err_op0, max_id_op0=max_id_op0, min_id_op0=min_id_op0,
            err_op1=err_op1, max_id_op1=max_id_op1, min_id_op1=min_id_op1,
            err_odd=err_odd, max_id_odd=max_id_odd, min_id_odd=min_id_odd,
            acc_dis=acc_dis, max_id_acc=max_id_acc, min_id_acc=min_id_acc,
            acc_a0_y0=group_acc[0][0], acc_a0_y1=group_acc[0][1], acc_a1_y0=group_acc[1][0],
            acc_a1_y1=group_acc[1][1]))

        # save result
        result = [args.save_name, epoch, domain_type, accs.avg, group_acc[0][0].item(),
                  group_acc[0][1].item(),
                  group_acc[1][0].item(), group_acc[1][1].item(), acc_var.item(), acc_dis.item(),
                  err_op0.item(),
                  err_op1.item(), err_odd.item()]

    return losses.avg, accs.avg, acc_var, err_odd, result


def eval_loop_consis(args, epoch, domain_type, dataloader, model):
    domain_msg = 'Source' if domain_type == 'source' else 'Target'

    # init statistics
    losses = AverageMeter()
    accs = AverageMeter()
    group_correct = torch.zeros((args.num_groups, args.num_labels))
    group_cnt = torch.zeros((args.num_groups, args.num_labels))
    consis_acc_per_group = AverageMeterVector(args.num_groups * args.num_labels)

    # switch to eval mode
    model.eval()

    # training criterion
    loss_fn = nn.CrossEntropyLoss()

    # dataloader
    iterator = enumerate(dataloader)

    # transform function (use for shapes only)
    transform_fn = transformation_function(args.dataset, args.transform_type)

    with torch.no_grad():
        for i, sample_batch in iterator:
            if args.dataset == 'utk-fairface':
                inputs, inputs_trans = sample_batch['image']
                inputs = inputs.to(args.device)
                inputs_trans = inputs_trans.to(args.device)
                labels = sample_batch['label']['gender'].to(args.device)
                groups = sample_batch['label']['race'].to(args.device)

            elif args.dataset == 'shapes':
                inputs = sample_batch[0].float().to(args.device)
                labels = sample_batch[1].long().squeeze().to(args.device)
                groups = sample_batch[2].long().squeeze().to(args.device)
                inputs_trans = transform_fn(inputs).to(args.device)

            elif args.dataset == 'newadult':
                inputs, inputs_trans = sample_batch[0]
                inputs = inputs.float().to(args.device)
                inputs_trans = inputs_trans.float().to(args.device)
                labels = sample_batch[1].long().squeeze().to(args.device)
                groups = sample_batch[2].long().squeeze().to(args.device)

            else:
                raise Exception(f'Unknown dataset: {args.dataset}')

            # forward
            outputs, features = model(inputs)

            loss = loss_fn(outputs, labels)
            losses.update(loss.item(), inputs.size(0))

            # statistics
            prec = accuracy(outputs, labels)[0]
            accs.update(prec.item(), inputs.size(0))
            batch_group_correct, batch_group_cnt = group_accuracy(args, outputs, labels, groups)
            group_correct += batch_group_correct
            group_cnt += batch_group_cnt

            # measure consistency (average case, not worst case)
            sub_consises = []
            sizes = []
            for g in range(args.num_groups):
                for y in range(args.num_labels):
                    sub_inputs = inputs[(groups == g) * (labels == y)]
                    sub_inputs_trans = inputs_trans[(groups == g) * (labels == y)]
                    if sub_inputs.size(0) > 0:
                        sub_consis = eval_consis(model, sub_inputs, sub_inputs_trans)
                        sub_consises.append(sub_consis.item())
                        sizes.append(sub_inputs.size(0))
                    else:
                        sub_consises.append(torch.tensor(0.0))
                        sizes.append(0)

            consis_acc_per_group.update(np.array(sub_consises), np.array(sizes))

        # measure unfairness
        acc_dis, max_id_acc, min_id_acc = acc_disparity(group_correct, group_cnt)
        group_acc = torch.nan_to_num(group_correct / group_cnt) * 100
        acc_var = torch.std(group_acc, unbiased=False)
        err_op0, max_id_op0, min_id_op0 = eql_op(group_acc, 0)
        err_op1, max_id_op1, min_id_op1 = eql_op(group_acc)
        err_odd, max_id_odd, min_id_odd = eql_odd(group_acc)

        # log
        print('Val {0} Epoch:{1} | Loss {loss:.4f} | Acc {acc:.2f} '
              '[{acc_a0_y0:.2f} {acc_a0_y1:.2f} {acc_a1_y0:.2f} {acc_a1_y1:.2f}]|'
              'Consis [{consis_a0y0:.2f} {consis_a0y1:.2f} {consis_a1y0:.2f} {consis_a1y1:.2f}]|'
              'acc_var {acc_var:.2f}|'
              'acc_dis {acc_dis:.2f}, ({max_id_acc}, {min_id_acc})|'
              'err_op_0 {err_op0:.2f}, ({max_id_op0}, {min_id_op0})|'
              'err_op_1 {err_op1:.2f}, ({max_id_op1}, {min_id_op1})|'
              'err_odd {err_odd:.2f}, ({max_id_odd}, {min_id_odd})|'.format(
            domain_msg, epoch, loss=losses.avg, acc=accs.avg, acc_var=acc_var,
            err_op0=err_op0, max_id_op0=max_id_op0, min_id_op0=min_id_op0,
            err_op1=err_op1, max_id_op1=max_id_op1, min_id_op1=min_id_op1,
            err_odd=err_odd, max_id_odd=max_id_odd, min_id_odd=min_id_odd,
            acc_dis=acc_dis, max_id_acc=max_id_acc, min_id_acc=min_id_acc,
            acc_a0_y0=group_acc[0][0], acc_a0_y1=group_acc[0][1], acc_a1_y0=group_acc[1][0],
            acc_a1_y1=group_acc[1][1],
            consis_a0y0=consis_acc_per_group.avg[0], consis_a0y1=consis_acc_per_group.avg[1],
            consis_a1y0=consis_acc_per_group.avg[2],
            consis_a1y1=consis_acc_per_group.avg[3]))

        # save result
        result = [args.save_name, epoch, domain_type, accs.avg, group_acc[0][0].item(),
                  group_acc[0][1].item(),
                  group_acc[1][0].item(),
                  group_acc[1][1].item(), consis_acc_per_group.avg[0], consis_acc_per_group.avg[1],
                  consis_acc_per_group.avg[2],
                  consis_acc_per_group.avg[3], acc_var.item(), acc_dis.item(), err_op0.item(),
                  err_op1.item(),
                  err_odd.item()]

    return losses.avg, accs.avg, acc_var, err_odd, result


def eval_consis(model, inputs, inputs_trans):
    model.eval()
    with torch.no_grad():
        outputs, _ = model(inputs)
        targets = torch.softmax(outputs.detach(), dim=-1)
        _, targets = torch.max(targets, dim=-1)
        outputs_trans, _ = model(inputs_trans)
        consis = accuracy(outputs_trans, targets)[0][0]
    return consis


def transformation_function(dataset, type):
    if dataset == 'utk-fairface':
        transform_fn = transforms.Compose([transforms.RandomHorizontalFlip()] +
                                          [transforms.RandomCrop(96, padding=8)]
                                          )
    elif dataset == 'shapes':
        if type == 'crop':
            rand_crop = randrange(56, 64)
            transform_fn = transforms.Compose(
                [transforms.CenterCrop(rand_crop), transforms.Resize([64, 64])])
        # elif type == 'randcrop':
        #     transform_fn = transforms.RandomCrop(size=64, padding=10, padding_mode='edge')
        # elif type == 'flip+crop':
        #     rand_crop = randrange(58, 64)
        #     transform_fn = transforms.Compose([transforms.CenterCrop(rand_crop),
        #                                        transforms.RandomHorizontalFlip(),
        #                                        transforms.Resize([64, 64])])
        # elif type == 'pad':
        #     rand = randrange(1, 20)
        #     transform_fn = transforms.Compose(
        #         [transforms.Pad(padding=[int(40 / rand), int(60 / rand), int(40 / rand), int(20 / rand)],
        #                         padding_mode='edge'),
        #          transforms.Resize([64, 64])])
        elif type == 'crop_pad':
            rand_crop = randrange(56, 64)
            rand_pad = randrange(1, 20)
            transform_fn = transforms.Compose(
                [transforms.Pad(
                    padding=[int(40 / rand_pad), int(60 / rand_pad), int(40 / rand_pad),
                             int(20 / rand_pad)],
                    padding_mode='edge'),
                    transforms.Resize([64, 64]),
                    transforms.CenterCrop(rand_crop),
                    transforms.Resize([64, 64])])
        else:
            raise Exception(f'Unknown transformation type: {type}')
    else:
        transform_fn = None

    return transform_fn


def load_data(args):
    if args.dataset == 'shapes':
        transform_train = transforms.ToTensor()

        transform = transforms.ToTensor()

        Y = "shape"
        A = "object_hue"
        D = "scale"
        y_binary = [0, 1]

        if args.shift_type == "Sshift1":
            YA_source = [0.1, 0.4, 0.4, 0.1]  # [Y0A0, Y0A1, Y1A0, Y1A1]
            YA_target = [0.1, 0.4, 0.4, 0.1]  # [Y0A0, Y0A1, Y1A0, Y1A1]
            D_dist_source = [4 / 16, 4 / 16, 3 / 16, 1 / 16, 1 / 16, 1 / 16, 1 / 16, 1 / 16]
            D_dist_target = [1 / 16, 1 / 16, 1 / 16, 1 / 16, 1 / 16, 3 / 16, 4 / 16, 4 / 16]
            size_s = 2000
            size_t = 5000
            train_proportion_s = 0.25
        elif args.shift_type == "Sshift2":
            YA_source = [0.1, 0.4, 0.4, 0.1]  # [Y0A0, Y0A1, Y1A0, Y1A1]
            YA_target = [0.4, 0.1, 0.1, 0.4]  # [Y0A0, Y0A1, Y1A0, Y1A1]
            D_dist_source = [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8]
            D_dist_target = [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8]
            size_s = 2000
            size_t = 5000
            train_proportion_s = 0.25
        elif args.shift_type == "Dshift":
            YA_source = [0.1, 0.4, 0.4, 0.1]  # [Y0A0, Y0A1, Y1A0, Y1A1]
            YA_target = [0.1, 0.4, 0.4, 0.1]  # [Y0A0, Y0A1, Y1A0, Y1A1]
            D_dist_source = [0.5, 0.5, 0., 0., 0., 0., 0., 0.]
            D_dist_target = [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8]
            size_s = 3000
            size_t = 5000
            train_proportion_s = 0.5
        elif args.shift_type == "Hshift":
            YA_source = [0.1, 0.4, 0.4, 0.1]  # [Y0A0, Y0A1, Y1A0, Y1A1]
            YA_target = [0.4, 0.1, 0.1, 0.4]  # [Y0A0, Y0A1, Y1A0, Y1A1]
            D_dist_source = [0.5, 0.5, 0., 0., 0., 0., 0., 0.]
            D_dist_target = [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8]
            size_s = 3000
            size_t = 5000
            train_proportion_s = 0.5
        else:
            raise Exception(f'Unknown shift type {args.shift_type}')

        s_train_dataset = data_loader.ShapesDataset(Y=Y, Y_binary=y_binary, A=A, A_binary=[0, 1],
                                                    Y0A0=YA_source[0],
                                                    Y0A1=YA_source[1], Y1A0=YA_source[2],
                                                    Y1A1=YA_source[3],
                                                    D=D, D_dist=D_dist_source,
                                                    data_path=args.data_root,
                                                    seed=args.seed, batch_size=args.batch_size,
                                                    phase='train', transform=transform_train,
                                                    size=size_s,
                                                    train_proportion=train_proportion_s)
        s_test_dataset = data_loader.ShapesDataset(Y=Y, Y_binary=y_binary, A=A, A_binary=[0, 1],
                                                   Y0A0=YA_source[0],
                                                   Y0A1=YA_source[1], Y1A0=YA_source[2],
                                                   Y1A1=YA_source[3],
                                                   D=D, D_dist=D_dist_source,
                                                   data_path=args.data_root,
                                                   seed=args.seed, batch_size=args.batch_size,
                                                   phase='test', transform=transform, size=size_s,
                                                   train_proportion=train_proportion_s)

        t_train_dataset = data_loader.ShapesDataset(Y=Y, Y_binary=y_binary, A=A, A_binary=[0, 1],
                                                    Y0A0=YA_target[0],
                                                    Y0A1=YA_target[1], Y1A0=YA_target[2],
                                                    Y1A1=YA_target[3],
                                                    D=D, D_dist=D_dist_target,
                                                    data_path=args.data_root,
                                                    seed=args.seed, batch_size=args.batch_size,
                                                    phase='train', transform=transform, size=size_t)
        t_test_dataset = data_loader.ShapesDataset(Y=Y, Y_binary=y_binary, A=A, A_binary=[0, 1],
                                                   Y0A0=YA_target[0],
                                                   Y0A1=YA_target[1], Y1A0=YA_target[2],
                                                   Y1A1=YA_target[3],
                                                   D=D, D_dist=D_dist_target,
                                                   data_path=args.data_root,
                                                   seed=args.seed, batch_size=args.batch_size,
                                                   phase='test', transform=transform, size=size_t)

    elif args.dataset == 'newadult':
        s_train_dataset = data_loader.NewAdultDataset(args.data_root, args.source_state, False,
                                                      '2018', task=args.state_task,
                                                      phase='train')
        s_test_dataset = data_loader.NewAdultDataset(args.data_root, args.source_state, False,
                                                     '2018', task=args.state_task,
                                                     phase='test')

        t_train_dataset = data_loader.NewAdultDataset(args.data_root, args.source_state, True,
                                                      '2018', task=args.state_task,
                                                      phase='train')
        t_test_dataset = data_loader.NewAdultDataset(args.data_root, args.source_state, True,
                                                     '2018', task=args.state_task,
                                                     phase='test')

    elif args.dataset == 'utk-fairface':
        transform = transforms.Compose([
            transforms.Resize(96),
            transforms.ToTensor()])
        if args.strong_trans:
            transform_weak = transforms.Compose([transforms.Resize(96),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.RandomCrop(96, padding=8),
                                                 transforms.ToTensor()]
                                                )
            transform_strong = transforms.Compose([transforms.Resize(96),
                                                   RandAugment(3, 5),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomCrop(96, padding=8),
                                                   transforms.ToTensor()]
                                                  )
        else:
            transform_weak = transform
            transform_strong = None

        # UTKFace data
        with open(os.path.join(args.data_root_utk, 'white_list.pkl'), 'rb') as f:
            utk_race1_list = pickle.load(f)
        with open(os.path.join(args.data_root_utk, 'black_list.pkl'), 'rb') as f:
            utk_race2_list = pickle.load(f)
        utk_train = utk_race1_list[0] + utk_race2_list[0]
        utk_test = utk_race1_list[1] + utk_race2_list[1]
        # fairface data
        ff_train = os.path.join(args.data_root_fairface, 'train_white_black.csv')
        ff_test = os.path.join(args.data_root_fairface, 'test_white_black.csv')

        # make data loaders
        s_train_dataset = data_loader.UTKFaceDataset(
            root=os.path.join(args.data_root_utk, 'UTKFace'),
            images_name_list=utk_train,
            transform=transform_weak, transform_strong=transform_strong)
        s_test_dataset = data_loader.UTKFaceDataset(
            root=os.path.join(args.data_root_utk, 'UTKFace'),
            images_name_list=utk_test,
            transform=transform, transform_strong=transform_strong)

        t_train_dataset = data_loader.FairFaceDataset(root=args.data_root_fairface,
                                                      images_file=ff_train,
                                                      transform=transform_weak,
                                                      transform_strong=transform_strong)
        t_test_dataset = data_loader.FairFaceDataset(root=args.data_root_fairface,
                                                     images_file=ff_test,
                                                     transform=transform,
                                                     transform_strong=transform_strong)

    else:
        raise Exception(f'Unknown dataset: {args.dataset}')

    return s_train_dataset, s_test_dataset, t_train_dataset, t_test_dataset


def load_model(args):
    if args.dataset == 'utk-fairface':
        if args.model == 'vgg16':
            features = models.vgg16(pretrained=False).features
            model = Face(features, args.num_labels)
        elif args.model == 'resnet18':
            model_ft = models.resnet18(pretrained=False)
            features = torch.nn.Sequential(*list(model_ft.children())[:-1])
            model = Face_Resnet(features, args.num_labels)
        else:
            raise Exception(f'Unknown model type {args.model}')
    elif args.dataset == 'newadult':
        if args.model == 'mlp':
            model = Adult_MLP(args.num_labels)
        else:
            raise Exception(f'Only support model type: mlp')
    elif args.dataset == 'shapes':
        if args.model == 'mlp':
            model = MLP(num_classes=args.num_labels)
        else:
            raise Exception(f'Only support model type: mlp')
    elif args.dataset == 'cifar10':
        backbone = models.resnet18(pretrained=args.pretrain)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        features = torch.nn.Sequential(*list(backbone.children())[:-1])
        model = Face_Resnet(features, args.num_labels)
    return model


def fill_args(args):
    if args.dataset == 'shapes':
        args.data_root = 'data/shapes/'
        args.save_path = 'checkpoint/shapes'
        args.save_csv_path = 'csv/shapes'
        args.batch_size = 128
        args.model = 'mlp'
        args.adv_hidden_dim = 64
        args.num_workers = 1
        args.reverse = True
        args.fair_weight = 2
        args.epoch = 200
        args.step_lr = 200
        if args.shift_type in ('Sshift1', 'Sshift2'):
            args.train_iteration = 4
            args.transform_type = 'crop'
            args.lr = 0.01
        elif args.shift_type in ('Dshift', 'Hshift'):
            args.train_iteration = 12
            args.transform_type = 'crop_pad'
            args.lr = 0.001

    elif args.dataset == 'newadult':
        args.data_root = 'data/newadult/'
        args.save_path = 'checkpoint/newadult'
        args.save_csv_path = 'csv/newadult'
        args.val_epoch = 2
        args.adv_hidden_dim = 128
        args.num_workers = 1
        args.epoch = 50
        args.model = 'mlp'
        args.source_state = ['CA']
        args.state_task = 'income'
        args.reverse = False
        args.fair_weight = 2
        args.batch_size = 1024
        args.train_iteration = 120
        args.step_lr = 50
        args.lr = 0.01
    elif args.dataset == 'utk-fairface':
        args.data_root_utk = 'data/UTKFace'
        args.data_root_fairface = 'data/fairface'
        args.save_path = 'checkpoint/utk-fairface'
        args.save_csv_path = 'csv/face'
        args.num_workers = 1
        args.reverse = False
    return args
