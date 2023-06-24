import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def fixmatch_loss(model, inputs, inputs_trans, args):
    if args.reverse:
        # use transformed input to predict pseudo labels
        temp = inputs
        inputs = inputs_trans
        inputs_trans = temp

    # pseudo label
    pseudo_outputs, _ = model(inputs)
    pseudo_label = torch.softmax(pseudo_outputs.detach(), dim=-1)
    max_probs, targets = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(args.threshold).float()
    logits, _ = model(inputs_trans)

    # consistency loss
    loss = (F.cross_entropy(logits, targets, reduction='none') * mask).mean()

    return loss


def fair_fixmatch_loss(model, inputs, inputs_trans, groups, args):
    if args.reverse:
        # use transformed input to predict pseudo labels
        temp = inputs
        inputs = inputs_trans
        inputs_trans = temp
    # pseudo label
    pseudo_outputs, _ = model(inputs)
    pseudo_label = torch.softmax(pseudo_outputs.detach(), dim=-1)
    max_probs, targets = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(args.threshold).float()
    logits, _ = model(inputs_trans)

    loss = torch.tensor(0.0).to(args.device)
    sub_losses = []
    sizes = []

    for a in range(args.num_groups):
        for y in range(args.num_labels):
            logits_subgroup = logits[(groups == int(a)) * (targets == int(y))]
            targets_subgroup = targets[(groups == int(a)) * (targets == int(y))]
            mask_subgroup = mask[(groups == int(a)) * (targets == int(y))]

            if logits_subgroup.size(0) > 0:
                sub_loss = (F.cross_entropy(logits_subgroup, targets_subgroup,
                                            reduction='none') * mask_subgroup).mean()
            else:
                sub_loss = torch.tensor(0.0)
            sub_losses.append(sub_loss)
            sizes.append(sum(mask_subgroup))

    if torch.count_nonzero(torch.tensor(sizes)) == args.num_groups * args.num_labels:
        # reweigh consistency loss of every group
        coefficients = [1 / i for i in sizes]
        coefficients = [i / sum(coefficients) for i in coefficients]
        loss += sum([i * j for i, j in zip(sub_losses, coefficients)])
    else:
        # if there are groups that have size of 0, use FixMatch loss to avoid 1/0
        loss = (F.cross_entropy(logits, targets, reduction='none') * mask).mean()

    return loss


def fair_fixmatch_loss_fix(model, inputs, inputs_trans, groups, args):
    if args.reverse:
        # use transformed input to predict pseudo labels
        temp = inputs
        inputs = inputs_trans
        inputs_trans = temp
    # pseudo label
    pseudo_outputs, _ = model(inputs)
    pseudo_label = torch.softmax(pseudo_outputs.detach(), dim=-1)
    max_probs, targets = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(args.threshold).float()
    logits, _ = model(inputs_trans)

    loss = torch.tensor(0.0).to(args.device)
    sub_losses = []
    sizes = []

    for a in range(args.num_groups):
        for y in range(args.num_labels):
            logits_subgroup = logits[(groups == int(a)) * (targets == int(y))]
            targets_subgroup = targets[(groups == int(a)) * (targets == int(y))]
            mask_subgroup = mask[(groups == int(a)) * (targets == int(y))]

            if logits_subgroup.size(0) > 0:
                sub_loss = (F.cross_entropy(logits_subgroup, targets_subgroup,
                                            reduction='none') * mask_subgroup).mean()
            else:
                sub_loss = torch.tensor(0.0)
            sub_losses.append(sub_loss)
            sizes.append(sum(mask_subgroup))

    if torch.count_nonzero(torch.tensor(sizes)) == args.num_groups * args.num_labels:
        # reweigh consistency loss of every group
        # coefficients = [1 / i for i in sizes]
        # coefficients = [i / sum(coefficients) for i in coefficients]
        coefficients = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
        loss += sum([i * j for i, j in zip(sub_losses, coefficients)])
    else:
        # if there are groups that have size of 0, use FixMatch loss to avoid 1/0
        loss = (F.cross_entropy(logits, targets, reduction='none') * mask).mean()

    return loss


def fixmatch_loss_cifar(model, inputs, inputs_aug, args):
    pseudo_outputs, _ = model(inputs)
    pseudo_label = torch.softmax(pseudo_outputs.detach(), dim=-1)
    max_probs, targets = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(args.threshold).float()
    inputs_trans = inputs_aug.to(args.device)
    logits, _ = model(inputs_trans)

    # consistency loss
    loss = (F.cross_entropy(logits, targets, reduction='none') * mask).mean()

    # sub-losses (not true loss, just for record)
    sub_losses = []
    sizes = []
    for y in range(args.num_labels):
        logits_subgroup = logits[targets == int(y)]
        targets_subgroup = targets[targets == int(y)]
        mask_subgroup = mask[targets == int(y)]
        if logits_subgroup.size(0) > 0:
            sub_loss = (F.cross_entropy(logits_subgroup, targets_subgroup,
                                        reduction='none') * mask_subgroup).mean()
        else:
            sub_loss = torch.tensor(0.0)
        sub_losses.append(sub_loss)
        sizes.append(sum(mask_subgroup))
    size = sum(sizes)

    return loss, size, sub_losses, sizes


def fair_fixmatch_loss_cifar(model, inputs, inputs_aug, args, pre_sizes):
    pseudo_outputs, _ = model(inputs)
    pseudo_label = torch.softmax(pseudo_outputs.detach(), dim=-1)
    max_probs, targets = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(args.threshold).float()
    inputs_trans = inputs_aug.to(args.device)
    logits, _ = model(inputs_trans)

    loss = torch.tensor(0.0).to(args.device)
    sub_losses = []
    sizes = []

    if torch.count_nonzero(pre_sizes) == 10:
        coefficient = 1 / pre_sizes.detach()
        coefficient = coefficient / coefficient.sum()
        loss_flag = True
    else:
        loss = (F.cross_entropy(logits, targets, reduction='none') * mask).mean()
        loss_flag = False

    # if labels == 'none':
    #     labels = targets
    for y in range(args.num_labels):
        logits_subgroup = logits[targets == int(y)]
        targets_subgroup = targets[targets == int(y)]
        mask_subgroup = mask[targets == int(y)]
        if logits_subgroup.size(0) > 0:
            sub_loss = (F.cross_entropy(logits_subgroup, targets_subgroup,
                                        reduction='none') * mask_subgroup).mean()
            if loss_flag:
                beta = coefficient[y]
                loss += sub_loss * beta
        else:
            sub_loss = torch.tensor(0.0)
        sub_losses.append(sub_loss)
        sizes.append(sum(mask_subgroup))
    size = sum(sizes)

    return loss, size, sub_losses, sizes


def kl_div_with_logit(q_logit, p_logit):
    ### return a matrix without mean over samples.
    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = (q * logq).sum(dim=1)
    qlogp = (q * logp).sum(dim=1)

    return qlogq - qlogp
