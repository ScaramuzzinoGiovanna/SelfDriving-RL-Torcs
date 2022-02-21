from environment import TorcsEnv
import utility
import os
import tensorflow as tf
import numpy as np
import argparse
from agent import Agent
from config_practice_race import Config_practice_race


def test(model):
    agent = Agent(test=True, model_name='test_' + folder)
    agent.initialize(test=True, local=model)
    vect_rewards = []
    vect_speeds = []
    dist_raced = []

    for i in range(1):
        speed_epis = []  # velocità per episodio per fare poi la media
        reward_epis = []
        total_info = []
        dist_raced_epis = 0
        done = False

        config.modify_initial_position()
        total_reward_ep = 0.
        obs = env.reset(relaunch=True)
        st_0 = np.array([obs['speedX'], obs['trackPos'], obs['angle']])

        if np.isnan(obs['angle']) or np.isnan(obs['speedX']) or np.isnan(obs['trackPos']):
            print(obs, st_0)
        while not done:
            action = agent.act(st_0, test=True)
            new_obs, reward, done, info = env.step(action, obs)
            st_1 = np.array([new_obs['speedX'], new_obs['trackPos'], new_obs['angle']])

            if info != 'error_nan' and info != 'server_down':
                total_reward_ep += reward
                speed_epis.append(st_1[0])
                reward_epis.append(reward)
                st_0 = st_1
            elif info == 'stuck':
                print('stuck')
                pass
            else:
                print('error: ', info, ' dist to center and to start: ', config.get_car_position())
                config.modify_initial_position()
                done = False
                obs = env.reset(relaunch=True)
                st_0 = np.array([obs['speedX'], obs['trackPos'], obs['angle']])
                total_reward_ep = 0.
                speed_epis = []
                reward_epis = []

            if done == True:
                total_info.append(info)
                dist_raced_epis = new_obs['distRaced']

        avg_speed_episode = sum(speed_epis) / len(speed_epis)
        print(" TEST  ------ TOTAL REWARD " + str(total_reward_ep) + "   AVG SPEED: " + str(
            avg_speed_episode) + '     DISTANCE RACED: ' + str(dist_raced_epis))
        print("")

        vect_rewards.append(total_reward_ep)
        vect_speeds.append(avg_speed_episode)
        dist_raced.append(dist_raced_epis)

    avg_reward = sum(vect_rewards) / len(vect_rewards)
    avg_speed = sum(vect_speeds) / len(vect_speeds)
    avg_dist_raced = sum(dist_raced) / len(dist_raced)
    return avg_speed, avg_reward, avg_dist_raced, total_info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-folder", "--folder_name", default='dqn_results')
    ap.add_argument("-v", "--vision", default=True)
    ap.add_argument("-p", "--port", default='3001')

    args = vars(ap.parse_args())

    folder = args['folder_name']
    vision = bool(args['vision'])
    port = int(args['port'])

    folder_tmp = os.path.join(folder, 'save_tmp')
    folder_plot = os.path.join(folder, 'plot')
    config = Config_practice_race()
    env = TorcsEnv(config.get_length_track(), vision=vision, port=port, test=True)

    path_results = os.path.join(folder, 'results.csv')
    model = tf.keras.models.load_model(os.path.join(folder_tmp, 'target.hdf5'))

    if vision == False:
        header = ['run', 'speed', 'reward', 'dist_raced', 'info']
        utility.create_csv(path_results, header)
        speed = []
        reward = []
        dist = []
        info = []

        for i in range(30):
            avg_speed, reward_,dist_raced, info = test(model)
            speed.append(avg_speed)
            reward.append(reward_)
            dist.append(dist_raced)
            utility.save_results(path_results, i, avg_speed, reward_, dist_raced, info)
        utility.plot_speed(speed, folder_plot, 'speed')
        utility.plot_reward(reward, folder_plot + 'plot/', 'reward')
        env.end()
    else:
        for i in range(30):
            avg_speed, reward_, dist_raced, info = test(model)




