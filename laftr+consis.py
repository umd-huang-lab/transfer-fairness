from argparse import ArgumentParser
from utils import *
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.dann import LaftrLoss, LaftrLoss_DP
from make_losses import *
import torch.multiprocessing
import torch.optim as optim

parser = ArgumentParser()
parser.add_argument('--dataset', choices=['shapes', 'newadult', 'utk-fairface'], default='utk-fairface')
parser.add_argument('--num-labels', type=int, default=2)
parser.add_argument('--num-groups', type=int, default=2)
parser.add_argument('--save-path', type=str, default='checkpoint')
parser.add_argument('--save-name', type=str, default='temp')
parser.add_argument('--save-model', action='store_true', default=False)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--test-batch-size', type=int, default=256)
parser.add_argument('--train-iteration', type=int, default=500, help='Number of iteration per epoch')
parser.add_argument('--normalize-loss', action='store_true', default=False)

parser.add_argument('--model', choices=['vgg16', 'resnet18', "mlp", 'cnn'], default='vgg16')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--step-lr', type=int, default=100)
parser.add_argument('--step-lr-gamma', type=float, default=0.1)
parser.add_argument('--val-epoch', type=int, default=1)
# parser.add_argument('--evl-consis', action='store_true', default=True)

parser.add_argument('--fair-type', choices=['dp', 'eql_op_0', 'eql_op_1', 'eql_odd'], default='eql_odd')
parser.add_argument('--adv-hidden-dim', type=int, default=128)
parser.add_argument('--fair-weight', type=float, default=1.)
parser.add_argument('--consis-weight-source', type=float, default=1.)
parser.add_argument('--consis-weight-target', type=float, default=1.)
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold for confidence')
parser.add_argument('--fair-consis', action='store_true', default=False,
                    help="Use the proposed fair consistency regularization or not")
parser.add_argument('--strong-trans', action='store_true', default=True,
                    help="Consistency regularization (FixMatch) uses strong augmentation and weak augmentation. Set True to enable such augmentation.")

args = parser.parse_args()


def main(args):
    # filling additional args for specific dataset
    args = fill_args(args)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(args)

    # load data
    s_train_dataset, s_test_dataset, t_train_dataset, t_test_dataset = load_data(args)

    print('Source dataset training size: {}'.format(len(s_train_dataset)))
    print('Source dataset test size: {}'.format(len(s_test_dataset)))
    print('Target dataset training size: {}'.format(len(t_train_dataset)))
    print('Target dataset test size: {}'.format(len(t_test_dataset)))

    s_train_dataloader = DataLoader(dataset=s_train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers)
    s_test_dataloader = DataLoader(dataset=s_test_dataset, batch_size=args.test_batch_size, shuffle=True,
                                   num_workers=args.num_workers)
    t_train_dataloader = DataLoader(dataset=t_train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers)
    t_test_dataloader = DataLoader(dataset=t_test_dataset, batch_size=args.test_batch_size, shuffle=True,
                                   num_workers=args.num_workers)

    loaders = (s_train_dataloader, s_test_dataloader, t_train_dataloader, t_test_dataloader)

    # make model
    model = load_model(args)
    print(model)
    model.to(args.device)
    model = train_model(args, model, loaders)

    return model


def train_model(args, model, loaders):
    s_train_dataloader, s_test_dataloader, t_train_dataloader, t_test_dataloader = loaders

    # fairness adversary
    group_discri = DomainDiscriminator(in_feature=model.z_dim, hidden_size=args.adv_hidden_dim).to(args.device)
    if args.fair_type == 'dp':
        fair_adv = LaftrLoss_DP(group_discri).to(args.device)
    else:
        fair_adv = LaftrLoss(group_discri).to(args.device)

    # optimizer
    # optimizer = optim.Adam((list(model.parameters()) + list(group_discri.parameters())), lr=args.lr,
    #                        weight_decay=args.weight_decay)
    optimizer = optim.SGD((list(model.parameters()) + list(group_discri.parameters())), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=args.step_lr_gamma)

    # statistic
    best_t_acc_fair_odd = 0
    best_t_acc_odd = 0
    best_t_unfair_odd = 0
    best_epoch_odd = 0
    best_t_acc_fair_var = 0
    best_t_acc_var = 0
    best_t_unfair_var = 0
    best_epoch_var = 0
    results = []

    for epoch in range(args.epoch):
        # train
        s_train_loss, s_train_prec, s_train_unfair = train_loop(args, epoch, (s_train_dataloader, t_train_dataloader),
                                                                model, fair_adv, optimizer)

        # validation
        if epoch % args.val_epoch == args.val_epoch - 1:
            # test in the source domain
            s_val_loss, s_val_prec, s_val_unfair_var, s_val_unfair_odd, s_result = eval_loop_consis(args, epoch, 'source', s_test_dataloader, model)
            results.append(s_result)

            # test in the target domain
            t_val_loss, t_val_prec, t_val_unfair_var, t_val_unfair_odd, t_result = eval_loop_consis(args, epoch, 'target', t_test_dataloader, model)
            results.append(t_result)

            # save best model according to performance on target test set (using acc_var as the unfairness metric)
            if t_val_prec - t_val_unfair_var >= best_t_acc_fair_var:
                best_t_acc_fair_var = t_val_prec - t_val_unfair_var
                best_t_acc_var = t_val_prec
                best_t_unfair_var = t_val_unfair_var
                best_epoch_var = epoch
                if args.save_model:
                    sd_info = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': (scheduler and scheduler.state_dict()),
                        'epoch': epoch
                    }
                    save_checkpoint(args, "best_var", sd_info)

            # save best model according to performance on target test set (using err_odd as the unfairness metric)
            if t_val_prec - t_val_unfair_odd >= best_t_acc_fair_odd:
                best_t_acc_fair_odd = t_val_prec - t_val_unfair_odd
                best_t_acc_odd = t_val_prec
                best_t_unfair_odd = t_val_unfair_odd
                best_epoch_odd = epoch
                if args.save_model:
                    sd_info = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': (scheduler and scheduler.state_dict()),
                        'epoch': epoch
                    }
                    save_checkpoint(args, "best_odd", sd_info)

        if scheduler: scheduler.step()

    # save results to csv
    fields = ["name", "epoch", "domain", "acc", "acc_A0Y0", "acc_A0Y1", "acc_A1Y0", "acc_A1Y1", "consis_A0Y0",
              "consis_A0Y1", "consis_A1Y0", "consis_A1Y1", "acc_var", "acc_dis", "err_op_0", "err_op_1",
              "err_odd"]

    with open(os.path.join(args.save_csv_path, args.save_name) + '.csv', 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerows(results)

    print('Done!')
    print('Best (odd) epoch: ', best_epoch_odd)
    print('Best (odd) target test acc: {acc:.4f}'.format(acc=best_t_acc_odd))
    print('Best (odd) target test unfairness: {acc:.4f}'.format(acc=best_t_unfair_odd))
    print('Best (var) epoch: ', best_epoch_var)
    print('Best (var) target test acc: {acc:.4f}'.format(acc=best_t_acc_var))
    print('Best (var) target test unfairness: {acc:.4f}'.format(acc=best_t_unfair_var))

    return model


def train_loop(args, epoch, dataloaders, model, fair_adv, optimizer):
    # init statistics
    cls_losses = AverageMeter()
    fair_losses = AverageMeter()
    consis_losses_s = AverageMeter()
    consis_losses_t = AverageMeter()
    accs = AverageMeter()
    fair_adv_accs = AverageMeter()
    group_correct = torch.zeros((args.num_groups, args.num_labels))
    group_cnt = torch.zeros((args.num_groups, args.num_labels))

    # switch to training mode
    model = model.train()

    # training criterion
    loss_fn = nn.CrossEntropyLoss()

    # prepare data loaders
    dataloader_s, dataloader_t = dataloaders

    # train
    for batch_idx in range(args.train_iteration):
        # load source data
        try:
            sample_batch_s = next(source_train_iter)
        except:
            source_train_iter = iter(dataloader_s)
            sample_batch_s = next(source_train_iter)
        if args.dataset == 'utk-fairface':
            inputs, inputs_strong = sample_batch_s['image']
            inputs = inputs.to(args.device)
            inputs_strong = inputs_strong.to(args.device)
            labels = sample_batch_s['label']['gender'].to(args.device)
            groups = sample_batch_s['label']['race'].to(args.device)

        elif args.dataset == 'adult' or args.dataset == 'shapes':
            inputs = sample_batch_s[0].float().to(args.device)
            labels = sample_batch_s[1].long().squeeze().to(args.device)
            groups = sample_batch_s[2].long().squeeze().to(args.device)

        # forward
        y, f = model(inputs)
        f_0 = f[(groups == 0)]
        f_1 = f[(groups == 1)]

        # main loss
        cls_loss = loss_fn(y, labels)
        loss = cls_loss

        # fairness loss
        if args.fair_type == 'dp':
            fair_loss = fair_adv(f_0, f_1)
        else:
            labels_a0 = labels[(groups == 0)]
            labels_a1 = labels[(groups == 1)]
            fair_loss = fair_adv(f_0, f_1, labels_a0, labels_a1, args.fair_type)
        loss = loss + args.fair_weight * fair_loss

        # statistics
        cls_losses.update(cls_loss.item(), inputs.size(0))
        prec = accuracy(y, labels)[0]
        accs.update(prec.item(), inputs.size(0))
        batch_group_correct, batch_group_cnt = group_accuracy(args, y, labels, groups)
        group_correct += batch_group_correct
        group_cnt += batch_group_cnt
        fair_losses.update(fair_loss.item(), inputs.size(0))
        fair_adv_acc = fair_adv.domain_discriminator_accuracy
        fair_adv_accs.update(fair_adv_acc.item(), inputs.size(0))

        # consistency regularization in source domain
        if args.fair_consis:
            loss_reg_s = fair_fixmatch_loss(model, inputs, inputs_strong, groups, args)
        else:
            loss_reg_s = fixmatch_loss(model, inputs, inputs_strong, args)
        consis_losses_s.update(loss_reg_s.item(), inputs.size(0))
        loss += args.consis_weight_source * loss_reg_s

        # consistency regularization in target domain
        # load target data
        try:
            sample_batch_t = next(target_train_iter)
        except:
            target_train_iter = iter(dataloader_t)
            sample_batch_t = next(target_train_iter)
        if args.dataset == 'utk-fairface':
            inputs_t, inputs_t_strong = sample_batch_t['image']
            inputs_t = inputs_t.to(args.device)
            inputs_t_strong = inputs_t_strong.to(args.device)
            # labels_t = sample_batch_t['label']['gender'].to(args.device)  # labels are not used in training
            groups_t = sample_batch_t['label']['race'].to(args.device)

        elif args.dataset == 'adult' or args.dataset == 'shapes':
            inputs_t = sample_batch_t[0].float().to(args.device)
            # labels_t = sample_batch_t[1].long().squeeze().to(args.device)
            groups_t = sample_batch_t[2].long().squeeze().to(args.device)

        if args.fair_consis:
            loss_reg_t = fair_fixmatch_loss(model, inputs_t, inputs_t_strong, groups_t, args)
        else:
            loss_reg_t = fixmatch_loss(model, inputs_t, inputs_t_strong, args)
        consis_losses_t.update(loss_reg_t.item(), inputs_t.size(0))
        loss += args.consis_weight_target * loss_reg_t

        if args.normalize_loss:
            loss = loss / (1 + args.fair_weight + args.consis_weight_source + args.consis_weight_target)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # measure unfairness in source domain
    acc_dis, max_id_acc, min_id_acc = acc_disparity(group_correct, group_cnt)
    group_acc = torch.nan_to_num(group_correct / group_cnt) * 100
    acc_var = torch.std(group_acc, unbiased=False)
    err_op0, max_id_op0, min_id_op0 = eql_op(group_acc, 0)
    err_op1, max_id_op1, min_id_op1 = eql_op(group_acc)
    err_odd, max_id_odd, min_id_odd = eql_odd(group_acc)

    # log
    print(
        '{0} | Cls_Loss:{loss_cls:.4f} Fair_Loss:{loss_fair:.4f} '
        'Consis_Loss_s:{loss_reg_s:.4f} Consis_Loss_t:{loss_reg_t:.4f} | '
        'Acc:{acc:.2f} FAdv_Acc:{fair_adv_acc:.2f} | Unfairness acc_var:{acc_var:.2f} '
        'acc_dis:{acc_dis:.2f} e_op_0:{err_op0:.2f} e_op_1:{err_op1:.2f} e_odd:{err_odd:.2f}'.format(
            epoch, loss_cls=cls_losses.avg, loss_fair=fair_losses.avg,
            loss_reg_s=consis_losses_s.avg, loss_reg_t=consis_losses_t.avg,
            acc=accs.avg, fair_adv_acc=fair_adv_accs.avg, acc_var=acc_var,
            acc_dis=acc_dis, err_op0=err_op0, err_op1=err_op1, err_odd=err_odd,
        ))

    return cls_losses.avg, accs.avg, err_odd


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    model = main(args)
