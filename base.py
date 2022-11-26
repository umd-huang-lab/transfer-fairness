"""
standard training on source data (option: data augmentation)
"""
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import data_loader
import pickle
from argparse import ArgumentParser
from make_models import *
from utils import *

parser = ArgumentParser()
parser.add_argument('--dataset', choices=['utk', 'adult', 'shapes', 'newadult', 'fairface', 'utk-fairface'],
                    default='utk')
parser.add_argument('--data-root', type=str, default='/data')
parser.add_argument('--meta-info', type=str, default='/data')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--target-batch-size', type=int, default=42)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--model', choices=['vgg11', 'vgg11_bn', 'vgg16', 'resnet18', "mlp", 'cnn'], default='vgg11')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--save-path', type=str, default='checkpoint')
parser.add_argument('--save-name', type=str, default='temp')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--step-lr', type=int, default=50)
parser.add_argument('--step-lr-gamma', type=float, default=0.1)
parser.add_argument('--augmentation', action='store_true', default=False)
parser.add_argument('--transforms',
                    choices=['none', 'crop', 'flip', 'jitter', 'rotate', 'cutout', 'blur', 'grayscale', 'poster',
                             'sharp', 'contrast']
                    , nargs='+', default='none')
parser.add_argument('--num-groups', type=int, default=2)
parser.add_argument('--num-labels', type=int, default=2)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--val-epoch', type=int, default=5)
parser.add_argument('--evl-consis', action='store_true', default=False)
parser.add_argument('--trans', type=str, default='none')

args = parser.parse_args()


def main(args):
    args = fill_args(args)
    print(args)

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load data
    s_train_dataset, s_test_dataset, t_train_dataset, t_test_dataset = load_data(args)

    print('source dataset train  size: {}'.format(len(s_train_dataset)))
    print('source dataset test size: {}'.format(len(s_test_dataset)))
    print('target dataset train size: {}'.format(len(t_train_dataset)))
    print('target dataset test size: {}'.format(len(t_test_dataset)))

    s_train_dataloader = DataLoader(dataset=s_train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers)
    s_test_dataloader = DataLoader(dataset=s_test_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers)
    t_train_dataloader = DataLoader(dataset=t_train_dataset, batch_size=args.target_batch_size, shuffle=True,
                                    num_workers=args.num_workers)
    t_test_dataloader = DataLoader(dataset=t_test_dataset, batch_size=args.batch_size, shuffle=True,
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
    optimizer, scheduler = make_optimizer(args, model)

    best_s_acc_fair = 0
    best_s_acc = 0
    best_s_unfair = 0
    best_epoch = 0

    for epoch in range(args.epoch):
        # train
        train_loss, train_prec = train_loop(args, epoch, 'source', s_train_dataloader, model, optimizer)

        # validation
        if epoch % args.val_epoch == args.val_epoch - 1:
            # source test
            s_val_loss, s_val_prec, s_val_unfair = eval_loop(args, epoch, 'source', s_test_dataloader, model)

            # target test
            t_val_loss, t_val_prec, t_val_unfair = eval_loop(args, epoch, 'target', t_test_dataloader, model)

            # save best model according to performance on source test set
            if s_val_prec - s_val_unfair >= best_s_acc_fair:
                best_s_acc_fair = s_val_prec - s_val_unfair
                best_s_acc = s_val_prec
                best_s_unfair = s_val_unfair
                best_epoch = epoch
                sd_info = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': (scheduler and scheduler.state_dict()),
                    'epoch': epoch
                }
                save_checkpoint(args, "best", sd_info)

        if scheduler: scheduler.step()

    # save last model
    sd_info = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': (scheduler and scheduler.state_dict()),
        'epoch': epoch
    }
    save_checkpoint(args, "last", sd_info)

    print('Finish!')
    print('Best epoch: ', best_epoch)
    print('Best source test acc: {acc:.4f}'.format(acc=best_s_acc * 100))
    print('Best source test unfairness: {acc:.4f}'.format(acc=best_s_unfair * 100))

    return model


def train_loop(args, epoch, domain_type, dataloader, model, optimizer):
    loop_msg = 'Train'
    domain_msg = 'Source' if domain_type == 'source' else 'Target'

    # init statistics
    losses = AverageMeter()
    acc = AverageMeter()
    group_correct = torch.zeros((args.num_groups, args.num_labels))
    group_cnt = torch.zeros((args.num_groups, args.num_labels))

    # switch to train mode
    model = model.train()

    # training criterion
    loss_fn = nn.CrossEntropyLoss()

    # iterator = tqdm(enumerate(dataloader), total=len(dataloader))
    iterator = enumerate(dataloader)

    for i, sample_batch in iterator:
        if args.dataset == 'utk' or args.dataset == 'utk-fairface':
            inputs = sample_batch['image'].to(args.device)
            labels = sample_batch['label']['gender'].to(args.device)
            groups = sample_batch['label']['race'].to(args.device)
            # 2 groups
            if args.num_groups == 2:
                groups[groups > 0] = 1
        elif args.dataset == 'adult' or args.dataset == 'shapes':
            inputs = sample_batch[0].float().to(args.device)
            labels = sample_batch[1].long().squeeze().to(args.device)
            groups = sample_batch[2].long().squeeze().to(args.device)

        # forward
        if args.augmentation:
            transform_fn = transformation_function(args.dataset, args.trans)
            inputs = transform_fn(inputs)
        outputs, _ = model(inputs)

        # main loss
        loss = loss_fn(outputs, labels)
        losses.update(loss.item(), inputs.size(0))

        batch_group_correct, batch_group_cnt = group_accuracy(args, outputs, labels, groups)
        group_correct += batch_group_correct
        group_cnt += batch_group_cnt

        prec = batch_group_correct.sum() / (batch_group_cnt.sum())
        acc.update(prec, inputs.size(0))

        acc_dis, max_id_acc, min_id_acc = acc_disparity(group_correct, group_cnt)
        group_acc = torch.nan_to_num(group_correct / group_cnt)
        err_op0, max_id_op0, min_id_op0 = eql_op(group_acc, 0)
        err_op1, max_id_op1, min_id_op1 = eql_op(group_acc)
        err_odd, max_id_odd, min_id_odd = eql_odd(group_acc)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('{0} {1} Epoch:{2} | Loss {loss:.4f} | Acc {acc:.2f}|| '
          'acc_dis {acc_dis:.2f}, ({max_id_acc}, {min_id_acc}) |'
          'err_op_0 {err_op0:.2f}, ({max_id_op0}, {min_id_op0}) |  '
          'err_op_1 {err_op1:.2f}, ({max_id_op1}, {min_id_op1}) |  '
          'err_odd {err_odd:.2f}, ({max_id_odd}, {min_id_odd}) ||'.format(
        domain_msg, loop_msg, epoch, loss=losses.avg, acc=acc.avg * 100,
        err_op0=err_op0 * 100, max_id_op0=max_id_op0, min_id_op0=min_id_op0,
        err_op1=err_op1 * 100, max_id_op1=max_id_op1, min_id_op1=min_id_op1,
        err_odd=err_odd * 100, max_id_odd=max_id_odd, min_id_odd=min_id_odd,
        acc_dis=acc_dis * 100, max_id_acc=max_id_acc, min_id_acc=min_id_acc))
    return losses.avg, acc.avg


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    model = main(args)
