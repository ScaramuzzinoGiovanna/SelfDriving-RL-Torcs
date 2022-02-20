import matplotlib.pyplot as plt
import csv
import os
import numpy as np
import pandas as p


def plot_reward(reward, folder_plot, file_name):
    plt.figure(figsize=(8, 6))
    plt.bar(range(0, len(reward)), reward, label='total reward')
    plt.xlabel("Episodes ")
    plt.ylabel("Rewards")
    plt.legend(loc="lower right")
    plt.savefig(folder_plot + file_name + '.png')


def plot_speed(speed, folder_plot, file_name):
    plt.figure(figsize=(8, 6))
    line_l = np.full(len(speed), 18)
    line_h = np.full(len(speed), 20)
    plt.bar(range(0, len(speed)), speed, label='avg_speed')
    plt.plot(range(0, len(speed)), line_l, linestyle='dashed', color='red')
    plt.plot(range(0, len(speed)), line_h, linestyle='dashed', color='red')
    plt.xlabel("Episodes (* 10^2)")
    plt.ylabel("Avg speeds")
    plt.legend(loc="lower right")
    axes = plt.gca()
    plt.savefig(folder_plot + file_name + '.png')


def plot_info(info, folder_plot, file_name):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(range(0, len(info)), info, label='info_stop')
    plt.title(file_name)
    plt.xlabel("Episodes")
    plt.ylabel("info")
    plt.legend(loc="lower right")
    axes = plt.gca()
    # axes.set_xlim([0, n_epochs - 1])
    # plt.xticks(np.arange(0, n_epochs, 1))
    plt.savefig(folder_plot + file_name + '.png')


def create_csv(file_csv, header):
    with open(file_csv, mode='a') as file:
        file_csv = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow(header)


def read_csv(path_file):
    df = p.read_csv(path_file)
    for i in range(2, 63):
        df['speed'][i]
        print(df)

def save_results(file_csv, epis, s, r, d, info):
    # create_csv(file_csv,header)
    with open(file_csv, mode='a') as file:
        file_csv = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow([epis, s, r, d, info])


def create_folder(name, path=None):
    if path == None:
        path_folder = name
    else:
        path_folder = os.path.join(path, name)
    try:
        os.makedirs(path_folder)
    except FileExistsError:
        print(' directory {} already exist'.format(path_folder))
        pass
    except OSError:
        print('creation of the directory {} failed'.format(path_folder))
        pass
    else:
        print("Succesfully created the directory {} ".format(path_folder))

    return path_folder


def show_and_save_results(folder_out, folder_plot, q_table, vision, total_reward, vect_avg_speed, vect_speed,
                          vect_avg_reward, count_meta_raggiunta):
    avg_reward = sum(total_reward) / len(total_reward)
    avg_speed = sum(vect_avg_speed) / len(vect_avg_speed)
    max_speed = max(vect_speed)
    max_avg_speed = max(vect_avg_speed)
    avg_reward_move = sum(vect_avg_reward) / len(vect_avg_reward)

    print('AVERAGE REWARD FOR MOVE', avg_reward_move)
    print('AVERAGE REWARD: ', avg_reward)
    print('AVERAGE SPEED: ', avg_speed)
    print('MAX SPEED: ', max_speed)
    print('MAX AVG SPEED', max_avg_speed)
    print('count meta ragg ', count_meta_raggiunta)
    print("")

    if vision == False:
        # np.savetxt(folder_out + 'avg_reward_move.csv', vect_avg_reward, delimiter=',')
        # np.savetxt(folder_out +'reward.csv', total_reward, delimiter=',')
        np.savetxt(folder_out + 'q_table.csv', q_table, delimiter=',')
        # np.savetxt(folder_out + 'avg_speed.csv', vect_avg_speed, delimiter=',')
        # save_results(folder_out + 'results.csv',
        #              ['avg_reward_move', 'avg_reward', 'avg_speed', 'max_speed', 'max_avg_speed',
        #               'count_raggiung_meta'],
        #              avg_reward_move, avg_reward, avg_speed, max_speed, max_avg_speed, count_meta_raggiunta)
        # plot_reward(total_reward, folder_plot, 'train_reward')
        # plot_speed(vect_avg_speed, folder_plot,'train_avg_speed')

    else:
        np.savetxt(folder_out + 'q_table_vision.csv', q_table, delimiter=',')

