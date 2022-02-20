import xml.etree.ElementTree as ET
import numpy as np


class Config_practice_race():
    def __init__(self, track_id=0):
        self.xmlfile = '/usr/local/share/games/torcs/config/raceman/practice.xml'
        self.tree = ET.parse(self.xmlfile)
        self.root = self.tree.getroot()
        self.dict_tracks = {'aalborg': 2587, 'e-track-3': 1621, 'e-track-5': 1621, 'eroad': 3260, 'dirt-4': 3260,
                            'alpine-1': 6355}
        self.category_tracks = {'aalborg': 'road', 'e-track-3': 'road', 'e-track-5': 'oval', 'eroad': 'road',
                                'dirt-4': 'dirt', 'alpine-1': 'road'}
        tracks = list(self.dict_tracks.keys())
        self.track_choosed = tracks[track_id]
        self.category_track_choosed = self.category_tracks[self.track_choosed]
        print('track choosed: ', self.track_choosed)

        self.set_default_file()

    def set_default_file(self):
        mod = False
        if self.root[1][1][0].attrib['val'] != self.track_choosed:
            mod = True
            self.root[1][1][0].attrib['val'] = self.track_choosed
            self.root[1][1][1].attrib['val'] = self.category_track_choosed

        if self.root[3][7][1].attrib['val'] != '100':
            mod = True
            self.root[3][7][1].attrib['val'] = ' 100'
        if self.root[3][7][0].attrib['val'] != '1':
            mod = True
            self.root[3][7][0].attrib['val'] = '1'
        if mod == True:
            self.tree.write(self.xmlfile)
            print(self.root[1][1][0].attrib['val'])
            print(self.root[1][1][1].attrib['val'])

    def get_length_track(self):
        # max_dist = self.find_track()
        max_dist = self.dict_tracks[self.track_choosed]
        return max_dist

    def set_track(self, name_track):
        print('scrivi metodo')

    def set_distance_to_start(self,
                              max_dist):  # 3 (practice),7 (Starting Grid),1 (dist to start)  #parte a tot metri dallo start
        self.root[3][7][1].attrib['val'] = str(np.random.randint(100, (
                max_dist - 100)))  # qui da valutare a seconda della lunghezza del percorso (max-100)

    def set_distance_to_center(self):
        self.root[3][7][0].attrib['val'] = str(np.random.randint(1, 12))  # 3 (practice),7 (Starting Grid), 0 (rows)

    def write_modif(self):
        self.tree.write(self.xmlfile)

    def get_distance_to_start(self):
        return self.root[3][7][1].attrib['val']

    def get_distance_to_center(self):
        return self.root[3][7][0].attrib['val']

    def modify_initial_position(self):
        max_dist = self.get_length_track()
        self.set_distance_to_center()
        self.set_distance_to_start(max_dist)
        self.write_modif()

    def get_car_position(self):
        dist_to_center = self.get_distance_to_center()
        dist_to_start = self.get_distance_to_start()
        return dist_to_center, dist_to_start
