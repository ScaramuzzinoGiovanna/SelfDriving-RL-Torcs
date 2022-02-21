from environment import TorcsEnv
from gym import spaces
import signal
import utility
import time
import tensorflow as tf
import numpy as np
import argparse
from agent import Agent
from keras import backend as K
import pickle
from config_practice_race import Config_practice_race


def keyboardInterruptHandler(signal=None, frame=None):
    agent.qnet_local.save(folder_tmp + 'local.hdf5')
    agent.qnet_target.save(folder_tmp + 'target.hdf5')
    buff,num_experiences= agent.get_buffer()
    with open(folder_tmp + 'replay_buffer.file', 'wb') as f:
        pickle.dump(buff, f, pickle.HIGHEST_PROTOCOL)
    print(num_experiences)
    np.savetxt(folder_tmp + 'num_exp.txt', num_experiences)

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
    ap.add_argument("-eps", "--epsilon", default='0.9')
    ap.add_argument("-c", "--episode_count", default='5000')
    ap.add_argument("-v", "--vision", default=False)
    ap.add_argument("-lr", "--learning_rate", default='0.0001')
    ap.add_argument("-folder", "--folder_name", default='dqn_results')
    ap.add_argument("-n_run", "--number_run", default='-1')

    args = vars(ap.parse_args())
    port = int(args['port'])
    episode_count = int(args['episode_count'])
    vision = bool(args['vision'])
    lr = float(args['learning_rate'])
    epsilon = float(args['epsilon'])
    folder_name = args['folder_name']
    number_run = int(args['number_run'])
    model_name = args['folder_name']
    done = False

    reward = 0
    discount = 0.95
    buffer_size = 100000
    min_buffer_size = 1000
    update_stats = 10
    K.tensorflow_backend._get_available_gpus()

    state_size = 3
    action_size = 9
    batch_size = 32

    folder_out = utility.create_folder('out/', folder_name)
    folder_model = utility.create_folder('models/', folder_name)
    folder_tmp = utility.create_folder('save_tmp/', folder_name)

    config = Config_practice_race()
    env = TorcsEnv(config.get_length_track(), vision=vision, port=port)
    agent = Agent(False, state_size, action_size, batch_size, buffer_size, min_buffer_size, lr, discount, model_name)
    if number_run == -1: # first run
        print('-1')
        agent.initialize()
    else:
        with open(folder_tmp + "replay_buffer.file", "rb") as f:
            buff = pickle.load(f)
        n_exp = np.genfromtxt(folder_tmp + "num_exp.txt")
        agent.initialize(test=False, run_intermedia=True, buff=buff, n_exp=n_exp, local=tf.keras.models.load_model(folder_tmp+'local.hdf5'), target=tf.keras.models.load_model(folder_tmp+'target.hdf5'))

    signal.signal(signal.SIGINT, keyboardInterruptHandler)
    total_reward = []
    vect_avg_speed = []  # velocità media per ogni episodio
    frame_idx = 0
    best_mean_reward = None
    start_time = time.time()
    for i in range((number_run+1), episode_count):
        done = False
        config.modify_initial_position()
        epsilon = 0.9 / (1 + 0.0005 * i)
        agent.tensorboard.step = i
        total_reward_ep = 0.

        obs = env.reset(relaunch=True)
        st_0 = np.array([obs['speedX'], obs['trackPos'], obs['angle']])
        if np.isnan(obs['angle']) or np.isnan(obs['speedX']) or np.isnan(obs['trackPos']):
            print(obs, st_0)
        speed_epis = []  # velocità per episodio per fare poi la media
        while not done:
            frame_idx += 1
            action = agent.act(st_0, epsilon)
            new_obs, reward, done, info = env.step(action, obs)
            total_reward.append(reward)
            mean_reward = np.mean(total_reward[-100:])
            try:
                st_1 = np.array([new_obs['speedX'], new_obs['trackPos'], new_obs['angle']])
            except TypeError:
                keyboardInterruptHandler()
                print(new_obs, new_obs['speedX'], new_obs['trackPos'], new_obs['angle'])
                raise
            if info != 'error_nan' and info != 'server_down':
                agent.step(st_0, action, reward, st_1, done)
                total_reward_ep += reward
                speed_epis.append(st_1[0])
                st_0 = st_1
            else:
                print('error: ', info, ' dist to center and to start: ', config.get_car_position())
                config.modify_initial_position()

                done = False
                obs = env.reset(relaunch=True)
                st_0 = np.array([obs['speedX'], obs['trackPos'], obs['angle']])
                total_reward_ep = 0.
                speed_epis = []

            if best_mean_reward is None or best_mean_reward < mean_reward:
                agent.qnet_local.save(
                    folder_model + f'best.hdf5')

        avg_speed_episode = sum(speed_epis) / len(speed_epis)
        max_speed_episode = max(speed_epis)
        min_speed_episode = min(speed_epis)
        vect_avg_speed.append(avg_speed_episode)
        dist_raced = obs['distRaced']
        agent.tensorboard.update_stats(distRacedEpis=dist_raced,total_reward_ep=total_reward_ep, avg_speed_episode=avg_speed_episode, max_speed_episode=max_speed_episode, min_speed_episode=min_speed_episode)
        # if not i % update_stats or i == 1:
        #     # average_reward = sum(total_reward[-update_stats:]) / len(total_reward[-update_stats:])
        #     # min_reward = min(total_reward[-update_stats:])
        #     # max_reward = max(total_reward[-update_stats:])
        #     # speed = sum(vect_avg_speed[-update_stats:]) / len(vect_avg_speed[-update_stats:])
        #     # agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon, speed_avg=speed)
        #     # vect_avg_speed = []
        #     # total_reward = []
        #     # agent.qnet_local.save(folder_model+f'{model_name}_epoch:{i}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min_{speed:_>7.2f}speed__{int(time.time())}.hdf5')
        #
        #     agent.qnet_local.save(folder_model + f'{model_name}_epoch:{i}__{total_reward_ep:_>7.2f}reward_{dist_raced:}dist_{avg_speed_episode:_>7.2f}avgSpeed__{int(time.time())}.hdf5')


        print(" episode  :  " + str(i) + " ------ TOTAL REWARD " + str(total_reward_ep) + "   AVG SPEED: " + str(
            avg_speed_episode) + '   EPSILON: ' + str(epsilon), 'distRaced: ' + str(obs['distRaced']))
        print('meta raggiunta n^: ', env.count_meta_raggiunta)
        print("")


    #utility.show_and_save_results(folder_out, folder_plot, q_table.get_q_table(), vision, total_reward, vect_avg_speed, vect_speed,
                                 # vect_avg_reward, env.count_meta_raggiunta)
    agent.qnet_local.save(
        folder_model + f'{model_name}_epoch:{i}__{total_reward_ep:_>7.2f}reward_{dist_raced:}dist_{avg_speed_episode:_>7.2f}avgSpeed__{int(time.time())}.hdf5')

    agent.qnet_local.save(folder_tmp + 'local_final.hdf5')
    agent.qnet_target.save(folder_tmp + 'target_final.hdf5')
    buff, num_experiences = agent.get_buffer()
    with open(folder_tmp + 'replay_buffer_final.file', 'wb') as f:
        pickle.dump(buff, f, pickle.HIGHEST_PROTOCOL)
    print(num_experiences)
    np.savetxt(folder_tmp + 'num_exp_final.txt', num_experiences)
    print("--- Execution in %s seconds ---" % (time.time() - start_time))
    env.end()
    print("Finish.")
