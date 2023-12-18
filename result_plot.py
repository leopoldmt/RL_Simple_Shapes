import numpy as np
import pandas as csv
import matplotlib.pyplot as plt
import os
import glob
from matplotlib import rcParams
from copy import copy


current_directory = os.getcwd()
PATH = '/results/inference/'

MODE = {'attr': ['attr'],
        'v': ['v'],
        'GWattr': ['GWattr', 'Gwv'],
        'GWv': ['GWattr', 'GWv'],
        'CLIPattr': ['CLIPattr', 'CLIPv'],
        'CLIPv': ['CLIPattr', 'CLIPv'],
        'GWsupattr': ['GWsupattr', 'GWsupv'],
        }

pos_mode = {'attr_from_attr': 0,
            'v_from_v': 1,

            'GWattr_from_GWattr': 3,
            'GWv_from_GWv': 4,

            'Gwv_from_GWattr': 6,
            'GWattr_from_GWv': 7,

            'CLIPattr_from_CLIPattr': 9,
            'CLIPv_from_CLIPv': 10,

            'CLIPv_from_CLIPattr': 12,
            'CLIPattr_from_CLIPv': 13,


            'GWsupattr_from_GWsupattr': 15,
            'GWsupv_from_GWsupattr': 16
            }

first_colors = {'attr_from_attr': '#1f77b4',
                'v_from_v': '#ff7f0e',

                'GWattr_from_GWattr': '#2ca02c',
                'GWv_from_GWv': '#d62728',

                'Gwv_from_GWattr': '#2ca02c',
                'GWattr_from_GWv': '#d62728',

                'CLIPattr_from_CLIPattr': '#9467bd',
                'CLIPv_from_CLIPv': '#8c564b',

                'CLIPv_from_CLIPattr': '#9467bd',
                'CLIPattr_from_CLIPv': '#8c564b',

                'GWsupattr_from_GWsupattr': '#e377c2',
                'GWsupv_from_GWsupattr': '#e377c2'}

second_colors = {'attr_from_attr': '#1f77b4',
                 'v_from_v': '#ff7f0e',

                 'GWattr_from_GWattr': '#2ca02c',
                 'GWv_from_GWv': '#d62728',

                 'Gwv_from_GWattr': '#d62728',
                 'GWattr_from_GWv': '#2ca02c',

                 'CLIPattr_from_CLIPattr': '#9467bd',
                 'CLIPv_from_CLIPv': '#8c564b',

                 'CLIPv_from_CLIPattr': '#8c564b',
                 'CLIPattr_from_CLIPv': '#9467bd',

                 'GWsupattr_from_GWsupattr': '#e377c2',
                 'GWsupv_from_GWsupattr': '#7f7f7f'}

if __name__ == '__main__':

    if PATH.split('/')[-2] == 'inference':
        fig, axs = plt.subplots(2, layout="constrained")
        j = 0
        folders = ['attr', 'v', 'GWattr', 'GWv', 'CLIPattr', 'CLIPv']
        # folders = ['GWattr']
        for folder in folders:
            path = 'results/inference/' + folder + '/'
            extension = 'npy'
            os.chdir(current_directory + '/' + path)
            result = glob.glob('*.{}'.format(extension))

            len_file = sorted(result)[:len(result) // 2]
            reward_file = sorted(result)[len(result) // 2:]

            sub_reward_file = [reward_file[:len(reward_file) // len(MODE[folder])], reward_file[len(reward_file) // len(MODE[folder]):]]
            sub_len_file = [len_file[:len(len_file) // len(MODE[folder])], len_file[len(len_file) // len(MODE[folder]):]]
            for num, mode_test in enumerate(MODE[folder]):
                name = mode_test + '_from_' + folder

                reward = np.zeros((len(sub_reward_file[num]), 1000))
                lenght = np.zeros((len(sub_len_file[num]), 1000))

                for i, file in enumerate(sub_reward_file[num]):
                    reward[i] = np.load(current_directory + '/results/inference/' + folder + '/' + file)
                for i, file in enumerate(sub_len_file[num]):
                    lenght[i] = np.load(current_directory + '/results/inference/' + folder + '/' + file)

                j = pos_mode[name]
                reward = reward.mean(axis=1)
                lenght = lenght.mean(axis=1)
                rcParams['hatch.linewidth'] = 2
                axs[0].bar(j, reward.mean(), yerr=reward.std(), label=name, facecolor=first_colors[name], edgecolor=second_colors[name], hatch='//')
                axs[0].set_title('Inference reward')
                axs[0].set_ylabel('Reward')
                axs[1].bar(j, lenght.mean(), yerr=reward.std(), label=name, facecolor=first_colors[name], edgecolor=second_colors[name], hatch='//')
                axs[1].set_title('Inference episode length')
                axs[1].set_ylabel('Episode length')
                # j += 1

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, borderaxespad=0.)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.15)
        # plt.show()
        plt.savefig("/home/leopold/Documents/Projets/Arena/RL/Simple_Shapes/results/inference/results.pdf", format="pdf")
        print('done')

    else:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        folders = ['attr', 'v', 'GWattr', 'GWv', 'CLIPattr']
        for folder in folders:
            path = 'results/' + folder + '/'
            extension = 'csv'
            os.chdir(current_directory + '/' + path)
            result = glob.glob('*.{}'.format(extension))
            len_file = sorted(result)[:len(result)//2]
            reward_file = sorted(result)[len(result)//2:]
            reward = np.zeros((len(reward_file), 611))
            lenght = np.zeros((len(len_file), 611))
            for i in range(len(len_file)):
                df_reward = csv.read_csv(current_directory + '/' + path + reward_file[i])
                df_len = csv.read_csv(current_directory + '/' + path + len_file[i])
                if len(df_reward) < 611:
                    reward[i, :len(df_reward)] = df_reward.iloc[:, 4]
                    lenght[i, :len(df_reward)] = df_len.iloc[:, 4]
                elif len(df_reward) > 611:
                    arrange_idx = np.arange(0, len(df_reward), len(df_reward)/611, dtype=int)
                    reward[i, :] = df_reward.iloc[arrange_idx, 4]
                    lenght[i, :] = df_len.iloc[arrange_idx, 4]
                else:
                    reward[i, :] = df_reward.iloc[:, 4]
                    lenght[i, :] = df_len.iloc[:, 4]
            reward[reward == 0] = np.nan
            lenght[lenght == 0] = np.nan
            reward_mean = np.nanmean(reward, axis=0)
            reward_std = np.nanstd(reward, axis=0)
            lenght_mean = np.nanmean(lenght, axis=0)
            lenght_std = np.nanstd(lenght, axis=0)
            axs[0].plot(np.arange(0, 611)*16384, lenght_mean, label=folder)
            axs[0].fill_between(np.arange(0, 611)*16384, lenght_mean - lenght_std, lenght_mean + lenght_std, alpha=0.2)
            axs[0].set_title('Episode length')
            axs[0].set_xlabel('Global step')
            axs[0].set_ylabel('Episode length')
            axs[0].legend(loc='upper right')
            axs[1].plot(np.arange(0, 611)*16384, reward_mean, label=folder)
            axs[1].fill_between(np.arange(0, 611)*16384, reward_mean - reward_std, reward_mean + reward_std, alpha=0.2)
            axs[1].set_title('Reward')
            axs[1].set_xlabel('Global step')
            axs[1].set_ylabel('Reward')
            axs[1].legend(loc='upper left')
        fig.tight_layout()
        plt.show()
        for file in sorted(result):
            div_file_name = file.split('_')
            df = csv.read_csv(PATH+file)
            if div_file_name[0] == 'len':
                axs[0].plot(df['global_step'], df.iloc[:, 4], label=div_file_name[1])
                axs[0].legend(loc='upper right')
                axs[0].set_title('Episode length')
                axs[0].set_xlabel('Global step')
                axs[0].set_ylabel('Episode length')
            elif div_file_name[0] == 'reward':
                axs[1].plot(df['global_step'], df.iloc[:, 4], label=div_file_name[1])
                axs[1].legend(loc='upper left')
                axs[1].set_title('Reward')
                axs[1].set_xlabel('Global step')
                axs[1].set_ylabel('Reward')
        fig.tight_layout()
        plt.show()