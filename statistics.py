from argparse import ArgumentParser
from utils import *

parser = ArgumentParser()
parser.add_argument('--dataset', choices=['shapes', 'newadult', 'utk-fairface'], default='utk-fairface')

args = parser.parse_args()

def main(args):
    args.strong_trans = False
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = fill_args(args)
    # load data
    s_train_dataset, s_test_dataset, t_train_dataset, t_test_dataset = load_data(args)
    print('Data loaded!')
    # groups_s_train = statistic(s_train_dataset, args)
    # print('groups_s_train:', groups_s_train)
    # groups_s_test = statistic(s_test_dataset, args)
    # print('groups_s_test:', groups_s_test)
    groups_t_train = statistic(t_train_dataset, args)
    print('groups_t_train:', groups_t_train)
    groups_t_test = statistic(t_test_dataset, args)
    print('groups_t_test:', groups_t_test)

if __name__ == "__main__":
    model = main(args)

