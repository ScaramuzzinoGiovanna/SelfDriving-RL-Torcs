from discretization import Discr
from environment import TorcsEnv
import numpy as np
from q_table import Q_table
import utility
from config_practice_race import Config_practice_race


def test(env, q_table, discr, c, n_avg_esp):
    vetc_rewards = []
    vect_speeds = []
    dist_raced = []

    for i in range(n_avg_esp):  # avg su stessa table
        done = False
        c.modify_initial_position()
        print(c.get_car_position())
        total_reward_ep = 0.
        initial_observation = env.reset(relaunch=True)
        obs = initial_observation
        obs_discr = discr.discr_state(initial_observation)  # return : [speed, angle, dist]
        speed_epis = []  # velocità per episodio per fare poi la media
        reward_epis = []
        total_info = []
        dist_raced_epis = 0

        while not done:

            action = q_table.get_max_action(obs_discr)
            new_obs, reward, done, info = env.step(action, obs_discr, obs)  # return discrete state

            if info != 'error_nan' and info != 'server_down' and info != 'stuck':
                new_obs_discr = discr.discr_state(new_obs)
                reward_epis.append(reward)
                total_reward_ep += reward
                speed_epis.append(new_obs['speedX'])
                reward_epis.append(reward)
                total_reward_ep += reward
                speed_epis.append(new_obs['speedX'])
                obs_discr = new_obs_discr
                obs = new_obs

            elif info == 'stuck':
                pass

            else:
                c.modify_initial_position()
                initial_observation = env.reset(relaunch=True)
                obs_discr = discr.discr_state(initial_observation)  # return : [speed, angle, dist]
                speed_epis = []  # velocità per episodio per fare poi la media
                reward_epis = []
                total_reward_ep = 0.
                done = False

            if done == True:
                total_info.append(info)
                dist_raced_epis = new_obs['distRaced']

        avg_speed_episode = sum(speed_epis) / len(speed_epis)
        print(" TEST  ------ TOTAL REWARD " + str(total_reward_ep) + "   AVG SPEED: " + str(
            avg_speed_episode) + '     DISTANCE RACED: ' + str(dist_raced_epis))
        print("")

        vetc_rewards.append(total_reward_ep)
        vect_speeds.append(avg_speed_episode)
        dist_raced.append(dist_raced_epis)

    avg_reward = sum(vetc_rewards) / len(vetc_rewards)
    avg_speed = sum(vect_speeds) / len(vect_speeds)
    avg_dist_raced = sum(dist_raced) / len(dist_raced)

    return avg_speed, avg_reward, avg_dist_raced, total_info


if __name__ == "__main__":
    path = 'results/'
    vision = False
    validation = False

    discr = Discr()

    n_avg_esp = 20
    every = 100

    if vision == False:
        header = ['run', 'speed', 'reward', 'dist_raced', 'info']
        speed = []
        reward = []
        dist = []
        info = []
        if validation:
            track_id = 0
            config = Config_practice_race(track_id)
            env = TorcsEnv(config.get_length_track(), vision=vision, port=3001, test=True)
            path_validation = path + 'validation/'
            path_res = path_validation + 'results_validation.csv'
            path_plot = path_validation + 'plot/'
            utility.create_folder(path_validation)
            utility.create_folder(path_plot)
            utility.create_csv(path_res, header)
            i = every
            while i != 17700:
                print(i)
                q = np.genfromtxt(path + 'out/q_table_' + str(i) + '.csv', delimiter=",")
                q_table = Q_table(discr.speed, discr.angle, discr.dist)
                q_table.inizialize_q_table(test=True, q_table=q)
                avg_speed, reward_, dist_raced, info = test(env, q_table, discr, config, n_avg_esp)
                speed.append(avg_speed)
                reward.append(reward_)
                dist.append(dist_raced)
                utility.save_results(path_res, i, avg_speed, reward_, dist_raced, info)
                i = i + every

            utility.plot_speed(speed, path_plot, 'speed')
            utility.plot_reward(reward, path_plot, 'reward')
            env.end()
        else:  # test different position e different track
            for track_id in range(5, 6):
                speed = []
                reward = []
                dist = []
                info = []
                config = Config_practice_race(track_id)
                max_lenght = 2587  # same lenght for all tacks
                env = TorcsEnv(max_lenght, vision=vision, port=3001, test=True)
                path_validation = path + 'test17500/'
                path_res = path_validation + 'results_test_track' + str(track_id) + '.csv'
                path_plot = path_validation + 'plot/' + 'track_' + str(track_id) + '/'
                utility.create_folder(path_validation)
                utility.create_folder(path_plot)
                n_avg_esp = 1
                q = np.genfromtxt(path + 'out/q_table_17500.csv', delimiter=",")
                q_table = Q_table(discr.speed, discr.angle, discr.dist)
                q_table.inizialize_q_table(test=True, q_table=q)
                utility.create_csv(path_res, header)
                for i in range(50):
                    avg_speed, reward_, dist_raced, info = test(env, q_table, discr, config, n_avg_esp)
                    speed.append(avg_speed)
                    reward.append(reward_)
                    dist.append(dist_raced)
                    utility.save_results(path_res, i, avg_speed, reward_, dist_raced, info)
                utility.plot_speed(speed, path_plot, 'speed')
                utility.plot_reward(reward, path_plot, 'reward')
                env.end()

    else:
        config = Config_practice_race(0)
        env = TorcsEnv(config.get_length_track(), vision=vision, port=3001, test=True)
        q = np.genfromtxt(path + 'out/q_table_17500.csv', delimiter=",")
        q_table = Q_table(discr.speed, discr.angle, discr.dist)
        q_table.inizialize_q_table(test=True, q_table=q)
        for i in range(1):
            avg_speed, reward, dist_raced, info = test(env, q_table, discr, config, n_avg_esp)
            i = i + 1
