from environment import TorcsEnv
from gym import spaces
from config_practice_race import Config_practice_race
import random
import signal
import utility
import time
from q_table import Q_table
from discretization import Discr
import numpy as np
import argparse


def keyboardInterruptHandler(signal, frame):
    if vision == False:
        np.savetxt(folder_out + 'q_table.csv', q_table.get_q_table(), delimiter=',')
        np.savetxt(folder_out + 'q_updates_' + str(i - 1) + '.csv', q_table.get_q_updates(), delimiter=',')
    print("--- Execution in %s seconds ---" % (time.time() - start_time))
    env.end()  # This is for shutting down TORCS
    print("")
    print("Finish.")
    print("call your function here".format(signal))
    exit(0)


def random_action():
    action = spaces.MultiDiscrete([3, 3]).sample()
    if action[0] == 2:
        action[0] = -1
    if action[1] == 2:
        action[1] = -1
    return action


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", default='3001')
    ap.add_argument("-c", "--episode_count", default='17500')
    ap.add_argument("-v", "--vision", default=False)
    ap.add_argument("-folder", "--folder_name", default='results')
    args = vars(ap.parse_args())
    port = int(args['port'])
    episode_count = int(args['episode_count'])
    vision = bool(args['vision'])
    folder_name = args['folder_name']
    done = False
    reward = 0
    discount = 0.95

    c = Config_practice_race()
    env = TorcsEnv(c.get_length_track(), vision=vision, port=3001, test=False)
    discr = Discr()
    q_table = Q_table(discr.speed, discr.angle, discr.dist)
    q_table.inizialize_q_table()

    dict_actions = q_table.dict_actions
    dict_states = q_table.dict_states

    folder_out = utility.create_folder('out/', folder_name)
    folder_plot = utility.create_folder('plot/', folder_name)

    signal.signal(signal.SIGINT, keyboardInterruptHandler)

    total_reward = []
    start_time = time.time()
    for i in range(episode_count):
        epsilon = 0.9 / (1 + 0.0005 * i)

        done = False
        c.modify_initial_position()

        total_reward_ep = 0.

        initial_observation = env.reset(relaunch=True)
        obs = initial_observation
        obs_discr = discr.discr_state(
            initial_observation)  # return : [speed, angle, dist]

        speed_epis = []  # velocità per episodio per fare poi la media
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random_action()
            else:
                action = q_table.get_max_action(obs_discr)
            new_obs, reward, done, info = env.step(action, obs_discr, obs)  # return discrete state
            if info != 'error_nan' and info != 'server_down':
                new_obs_discr = discr.discr_state(new_obs)
                old_value = q_table.get_q_value(obs_discr, action)
                next_max = q_table.get_max_q_value(new_obs_discr)
                count_q_update = q_table.get_updates(obs_discr, action)
                lr = max(np.round((0.5 / (1 + (0.05 * count_q_update))), 5), 0.0001)
                new_value = (1 - lr) * old_value + lr * (reward + discount * next_max)
                q_table.set_q_value(obs_discr, action, new_value)

                total_reward_ep += reward
                speed_epis.append(new_obs['speedX'])  # memorizzo la velocità non discretizzata di un episodio

                obs_discr = new_obs_discr
                obs = new_obs
            else:
                print('error: ', info, ' dist to center and to start: ', c.get_car_position())
                done = False
                c.modify_initial_position()
                initial_observation = env.reset(relaunch=True)
                total_reward_ep = 0.
                obs = initial_observation
                obs_discr = discr.discr_state(initial_observation)  # return : [speed, angle, dist]
                speed_epis = []
                # reward_epis = []

        avg_speed_episode = sum(speed_epis) / len(speed_epis)
        total_reward.append(total_reward_ep)
        print(" episode  :  " + str(i) + " ------ TOTAL REWARD " + str(total_reward_ep) + "   AVG SPEED: " + str(
            avg_speed_episode) + ' MAX SPEED ' + str(max(speed_epis)) + '   EPSILON: ' + str(epsilon))
        print("")

        if i % 10 == 0 and i != 0:  # test
            np.savetxt(folder_out + 'q_table_' + str(i) + '.csv', q_table.get_q_table(), delimiter=',')

    if vision == False:
        np.savetxt(folder_out + 'q_table.csv', q_table.get_q_table(), delimiter=',')
        np.savetxt(folder_out + 'q_updates.csv', q_table.get_q_updates(), delimiter=',')

    print("--- Execution in %s seconds ---" % (time.time() - start_time))
    env.end()
    print("Finish.")
