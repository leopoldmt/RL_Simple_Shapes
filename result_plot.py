import numpy as np
import pandas as csv
import matplotlib.pyplot as plt
import os
import glob

current_directory = os.getcwd()
PATH = '/results/inference/'

MODE = {'attr': ['attr'],
        'v': ['v'],
        'GWattr': ['GWattr', 'Gwv'],
        'GWv': ['GWattr', 'GWv']
        }

pos_mode = {'attr_from_attr': 0, 'v_from_v': 1, 'GWattr_from_GWattr': 3, 'GWv_from_GWv': 4, 'GWattr_from_GWv': 6, 'Gwv_from_GWattr': 7}

if __name__ == '__main__':

    if PATH.split('/')[-2] == 'inference':
        fig, axs = plt.subplots(2, layout="constrained")
        j = 0
        folders = ['attr', 'v', 'GWattr', 'GWv']
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
                axs[0].bar(j, reward.mean(), yerr=reward.std(), label=name)
                axs[0].set_title('Inference reward')
                axs[0].set_ylabel('Reward')
                axs[1].bar(j, lenght.mean(), yerr=lenght.std(), label=name)
                axs[1].set_title('Inference episode length')
                axs[1].set_ylabel('Episode length')
                # j += 1

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, borderaxespad=0.)
        fig.tight_layout()
        plt.show()
        print('done')

    else:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        folders = ['attr', 'v', 'GWattr', 'GWv']
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