import snakeoil3_gym as snakeoil3
import numpy as np
import os
import time


class TorcsEnv:
    initial_reset = True

    def __init__(self, len_track, vision=False, test=False, port=3001):
        self.vision = vision
        self.p = port
        self.test = test
        self.length_track = len_track
        self.count_meta_raggiunta = 0
        self.count = 0
        self.stuck_count = 0
        self.speed_prec = 0

        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime -vision &')
            time.sleep(0.5)
            os.system('sh autostart.sh')
            time.sleep(0.5)
        else:
            os.system(
                'torcs -r /usr/local/share/games/torcs/config/raceman/practice.xml -nofuel -nodamage -nolaptime -p {}&'.format(
                    self.p))

    def step(self, action, obs):

        self.obs = obs

        client = self.client
        acc = action[0]
        steer = action[1]

        R = client.R.d
        if acc == -1:
            R['accel'] = 0
            R['brake'] = 1
        elif acc == 0:
            R['accel'] = 0
            R['brake'] = 0
        elif acc == 1:
            R['accel'] = 1
            R['brake'] = 0
        R['steer'] = steer

        info = {}
        client.respond_to_server()
        code = client.get_servers_input()
        if code == -1:
            # client.R.d['meta'] = True
            print('Terminating because server stopped responding')
            return None, 0, client.R.d['meta'], 'server_down'

        new_obs = client.S.d
        if new_obs['speedX'] == None or new_obs['trackPos'] == None or new_obs['angle'] == None:
            print(new_obs)
            return None, 0, False, 'error_nan'
        if np.isnan(obs['angle']) or np.isnan(obs['speedX']) or np.isnan(obs['trackPos']) or np.isnan(
                new_obs['angle']) or np.isnan(new_obs['speedX']) or np.isnan(new_obs['trackPos']):
            return None, 0, False, 'error_nan'

        if self.test == True:
            if round(obs['speedX']) == 0 and self.speed_prec == 0:
                self.stuck_count += 1
                if self.stuck_count == 2000:
                    self.client.R.d['meta'] = True
                    self.stuck_count = 0
                    return obs, -1000, True, 'stuck'
            elif round(obs['speedX']) == 0 and self.speed_prec != 0:
                self.stuck_count = 0
            self.speed_prec = round(obs['speedX'])

        if obs['damage'] > 0 or np.cos(obs['angle']) < 0 or obs['trackPos'] <= -1 or obs['trackPos'] >= 1 or (
                np.abs((np.array(obs['track']))).any()) > 1:
            self.client.R.d['meta'] = True
            print('stop: wrong behaviour ----- Distance Raced: ' + str(obs['distRaced']) + ' su ' + str(
                self.length_track) + ' km')
            info = 'out track'

        elif obs['speedX'] <= -0:  # l'auto torna indietro #-1
            reward = -15
        else:
            reward = self.calculate_reward(np.array([obs['speedX'], obs['trackPos'], obs['angle']]))

        if self.length_track - 1 <= obs['distRaced'] and self.length_track + 1 >= obs['distRaced']:
            self.client.R.d['meta'] = True
            self.count_meta_raggiunta = self.count_meta_raggiunta + 1
            print('meta raggiunta - giro completo')

            info = 'meta raced'

        if client.R.d['meta'] is True:  # Send a reset signal
            client.respond_to_server()

        return new_obs, reward, client.R.d['meta'], info

    def calculate_reward(self, state):
        if np.abs(state[0]) < 18:
            r_speed = (1 / 20) * np.abs(state[0])
        elif np.abs(state[0]) >= 18 and np.abs(state[0]) <= 20:
            r_speed = 1
        elif np.abs(state[0]) > 20 and np.abs(state[0]) <= 40:
            r_speed = 2 - (np.abs(state[0]) / 20)
        elif np.abs(state[0]) > 40:
            r_speed = 0
        r_dist = np.abs(state[1])
        if np.abs(state[2]) >= 0.3:
            r_angle = 1
        elif np.abs(state[2]) <= 0.05:
            r_angle = 0
        else:
            r_angle = (1 / (0.3)) * np.abs(state[2])
        reward = r_speed - r_dist - r_angle

        if reward == None:
            print('reward None', state)
        return reward

    def reset(self, relaunch=False):
        # print("Reset")
        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            if relaunch is True:
                self.reset_torcs()
                # print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=self.p, vision=self.vision)  # Open new UDP in vtorcs
        self.client.maxSteps = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.dist_raced = obs['distRaced']
        self.initial_reset = False
        return obs

    def end(self):
        os.system('pkill torcs')

    def get_obs(self):  # return observation non discr
        return self.obs

    def reset_torcs(self):
        # print("relaunch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime -vision &')
            time.sleep(0.5)
            os.system('sh autostart.sh')
            time.sleep(0.5)
        else:
            os.system(
                'torcs -r /usr/local/share/games/torcs/config/raceman/practice.xml -nofuel -nodamage -nolaptime &')
