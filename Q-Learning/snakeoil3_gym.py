#!/usr/bin/python

# snakeoil.py
# Chris X Edwards <snakeoil@xed.ch>
# Snake Oil is a Python library for interfacing with a TORCS
# race car simulator which has been patched with the server
# extentions used in the Simulated Car Racing competitions.
# http://scr.geccocompetitions.com/
#
# To use it, you must import it and create a "drive()" function.
# This will take care of option handling and server connecting, etc.
# To see how to write your own client do something like this which is
# a complete working client:
# /-----------------------------------------------\
# |#!/usr/bin/python                              |
# |import snakeoil                                |
# |if __name__ == "__main__":                     |
# |    C= snakeoil.Client()                       |
# |    for step in xrange(C.maxSteps,0,-1):       |
# |        C.get_servers_input()                  |
# |        snakeoil.drive_example(C)              |
# |        C.respond_to_server()                  |
# |    C.shutdown()                               |
# \-----------------------------------------------/
# This should then be a full featured client. The next step is to
# replace 'snakeoil.drive_example()' with your own. There is a
# dictionary which holds various option values (see `default_options`
# variable for all the details) but you probably only need a few
# things from it. Mainly the `trackname` and `stage` are important
# when developing a strategic bot.
#
# This dictionary also contains a ServerState object
# (key=S) and a DriverAction object (key=R for response). This allows
# you to get at all the information sent by the server and to easily
# formulate your reply. These objects contain a member dictionary "d"
# (for data dictionary) which contain key value pairs based on the
# server's syntax. Therefore, you can read the following:
#    angle, curLapTime, damage, distFromStart, distRaced, focus,
#    fuel, gear, lastLapTime, opponents, racePos, rpm,
#    speedX, speedY, speedZ, track, trackPos, wheelSpinVel, z
# The syntax specifically would be something like:
#    X= o[S.d['tracPos']]
# And you can set the following:
#    accel, brake, clutch, gear, steer, focus, meta
# The syntax is:
#     o[R.d['steer']]= X
# Note that it is 'steer' and not 'steering' as described in the manual!
# All values should be sensible for their type, including lists being lists.
# See the SCR manual or http://xed.ch/help/torcs.html for details.
#
# If you just run the snakeoil.py base library itself it will implement a
# serviceable client with a demonstration drive function that is
# sufficient for getting around most tracks.
# Try `snakeoil.py --help` to get started.

# for Python3-based torcs python robot client
import socket
import sys
import getopt
import os
import time

PI = 3.14159265359

data_size = 2 ** 17


def clip(v, lo, hi):
    if v < lo:
        return lo
    elif v > hi:
        return hi
    else:
        return v


def bargraph(x, mn, mx, w, c='X'):
    '''Draws a simple asciiart bar graph. Very handy for
    visualizing what's going on with the data.
    x= Value from sensor, mn= minimum plottable value,
    mx= maximum plottable value, w= width of plot in chars,
    c= the character to plot with.'''
    if not w: return ''  # No width!
    if x < mn: x = mn  # Clip to bounds.
    if x > mx: x = mx  # Clip to bounds.
    tx = mx - mn  # Total real units possible to show on graph.
    if tx <= 0: return 'backwards'  # Stupid bounds.
    upw = tx / float(w)  # X Units per output char width.
    if upw <= 0: return 'what?'  # Don't let this happen.
    negpu, pospu, negnonpu, posnonpu = 0, 0, 0, 0
    if mn < 0:  # Then there is a negative part to graph.
        if x < 0:  # And the plot is on the negative side.
            negpu = -x + min(0, mx)
            negnonpu = -mn + x
        else:  # Plot is on pos. Neg side is empty.
            negnonpu = -mn + min(0, mx)  # But still show some empty neg.
    if mx > 0:  # There is a positive part to the graph
        if x > 0:  # And the plot is on the positive side.
            pospu = x - max(0, mn)
            posnonpu = mx - x
        else:  # Plot is on neg. Pos side is empty.
            posnonpu = mx - max(0, mn)  # But still show some empty pos.
    nnc = int(negnonpu / upw) * '-'
    npc = int(negpu / upw) * c
    ppc = int(pospu / upw) * c
    pnc = int(posnonpu / upw) * '_'
    return '[%s]' % (nnc + npc + ppc + pnc)


class Client():
    def __init__(self, H=None, p=None, i=None, e=None, t=None, s=None, d=None, vision=False):
        # If you don't like the option defaults,  change them here.
        self.vision = vision

        self.host = 'localhost'
        self.port = p
        self.sid = 'SCR'
        self.maxEpisodes = 1  # "Maximum number of learning episodes to perform"
        self.trackname = 'unknown'
        self.stage = 3  # 0=Warm-up, 1=Qualifying 2=Race, 3=unknown <Default=3>
        self.debug = False
        self.maxSteps = 100000  # 50steps/second
        self.S = ServerState()
        self.R = DriverAction()
        self.setup_connection()

    def setup_connection(self):
        # == Set Up UDP Socket ==
        try:
            self.so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as emsg:
            print('Error: Could not create socket...')
            sys.exit(-1)
        # == Initialize Connection To Server ==
        self.so.settimeout(1)

        n_fail = 5
        while True:
            # This string establishes track sensor angles! You can customize them.
            # a= "-90 -75 -60 -45 -30 -20 -15 -10 -5 0 5 10 15 20 30 45 60 75 90"
            # xed- Going to try something a bit more aggressive...
            a = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"

            initmsg = '%s(init %s)' % (self.sid, a)

            try:
                self.so.sendto(initmsg.encode(), (self.host, self.port))
            except socket.error as emsg:
                sys.exit(-1)
            sockdata = str()
            try:
                sockdata, addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                # print("Waiting for server on port %d............" % self.port)
                # print("Count Down : " + str(n_fail))
                if n_fail < 0:
                    # print("relaunch torcs")
                    os.system('pkill torcs')
                    time.sleep(1.0)
                    if self.vision is False:
                        os.system(
                            'torcs -r /usr/local/share/games/torcs/config/raceman/practice.xml -nofuel -nodamage -nolaptime &')
                    else:
                        os.system('torcs -nofuel -nodamage -nolaptime -vision &')
                        time.sleep(1.0)
                        os.system('sh autostart.sh')

                    n_fail = 5
                n_fail -= 1

            identify = '***identified***'
            if identify in sockdata:
                # print("Client connected on %d.............." % self.port)
                break

    def get_servers_input(self):
        '''Server's input is stored in a ServerState object'''
        if not self.so: return
        sockdata = str()
        n_fail = 5
        n_fail_org = n_fail
        while True:
            try:
                # Receive server data
                sockdata, addr = self.so.recvfrom(data_size)
                sockdata = sockdata.decode('utf-8')
            except socket.error as emsg:
                # pass
                # print('.', end=' ')
                # print "Waiting for data on %d.............." % self.port
                if n_fail < 0:
                    # self.shutdown()
                    return -1
                    n_fail = n_fail_org

                n_fail -= 1

            if '***identified***' in sockdata:
                pass
                # print("Client connected on %d.............." % self.port)
                continue
            elif '***shutdown***' in sockdata:
                print((("Server has stopped the race on %d. " +
                        "You were in %d place.") %
                       (self.port, self.S.d['racePos'])))
                self.shutdown()
                return
            elif '***restart***' in sockdata:
                # What do I do here?
                # print("Server has restarted the race on %d." % self.port)
                # I haven't actually caught the server doing this.
                self.shutdown()
                return
            elif not sockdata:  # Empty?
                continue  # Try again.
            else:
                self.S.parse_server_str(sockdata)
                if self.debug:
                    sys.stderr.write("\x1b[2J\x1b[H")  # Clear for steady output.
                    print(self.S)
                break  # Can now return from this function.

    def respond_to_server(self):
        if not self.so: return
        try:
            message = repr(self.R)
            self.so.sendto(message.encode(), (self.host, self.port))
        except socket.error as emsg:
            print("Error sending to server: %s Message %s" % (emsg[1], str(emsg[0])))
            sys.exit(-1)
        if self.debug: print(self.R.fancyout())
        # Or use this for plain output:
        # if self.debug: print self.R

    def shutdown(self):
        if not self.so: return
        print(("Race terminated or %s steps elapsed. Shutting down %d."
               % (str(self.maxSteps), self.port)))
        self.so.close()
        self.so = None
        # sys.exit() # No need for this really.


class ServerState():
    '''What the server is reporting right now.'''

    def __init__(self):
        self.servstr = str()
        self.d = dict()

    def parse_server_str(self, server_string):
        '''Parse the server string.'''
        self.servstr = server_string.strip()[:-1]
        sslisted = self.servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w = i.split(' ')
            self.d[w[0]] = destringify(w[1:])

    def __repr__(self):
        # Comment the next line for raw output:
        return self.fancyout()
        # -------------------------------------
        out = str()
        for k in sorted(self.d):
            strout = str(self.d[k])
            if type(self.d[k]) is list:
                strlist = [str(i) for i in self.d[k]]
                strout = ', '.join(strlist)
            out += "%s: %s\n" % (k, strout)
        return out

    def fancyout(self):
        '''Specialty output for useful ServerState monitoring.'''
        out = str()
        sensors = [  # Select the ones you want in the order you want them.
            # 'curLapTime',
            # 'lastLapTime',
            # 'stucktimer',
            # 'damage',
            # 'focus',
            # 'fuel',
            # 'gear',
            'distRaced',
            'distFromStart',
            # 'racePos',
            # 'opponents',
            # 'wheelSpinVel',
            # 'z',
            # 'speedZ',
            # 'speedY',
            'speedX',
            # 'targetSpeed',
            # 'rpm',
            # 'skid',
            # 'slip',
            # 'track',
            'trackPos',
            'angle',
        ]

        # for k in sorted(self.d): # Use this to get all sensors.
        for k in sensors:
            if type(self.d.get(k)) is list:  # Handle list type data.
                if k == 'track':  # Nice display for track sensors.
                    strout = str()
                    raw_tsens = ['%.1f' % x for x in self.d['track']]
                    strout += ' '.join(raw_tsens[:9]) + '_' + raw_tsens[9] + '_' + ' '.join(raw_tsens[10:])
                elif k == 'opponents':  # Nice display for opponent sensors.
                    strout = str()
                    for osensor in self.d['opponents']:
                        if osensor > 190:
                            oc = '_'
                        elif osensor > 90:
                            oc = '.'
                        elif osensor > 39:
                            oc = chr(int(osensor / 2) + 97 - 19)
                        elif osensor > 13:
                            oc = chr(int(osensor) + 65 - 13)
                        elif osensor > 3:
                            oc = chr(int(osensor) + 48 - 3)
                        else:
                            oc = '?'
                        strout += oc
                    strout = ' -> ' + strout[:18] + ' ' + strout[18:] + ' <-'
                else:
                    strlist = [str(i) for i in self.d[k]]
                    strout = ', '.join(strlist)
            else:  # Not a list type of value.
                if k == 'gear':  # This is redundant now since it's part of RPM.
                    gs = '_._._._._._._._._'
                    p = int(self.d['gear']) * 2 + 2  # Position
                    l = '%d' % self.d['gear']  # Label
                    if l == '-1': l = 'R'
                    if l == '0':  l = 'N'
                    strout = gs[:p] + '(%s)' % l + gs[p + 3:]
                elif k == 'damage':
                    strout = '%6.0f %s' % (self.d[k], bargraph(self.d[k], 0, 10000, 50, '~'))
                elif k == 'fuel':
                    strout = '%6.0f %s' % (self.d[k], bargraph(self.d[k], 0, 100, 50, 'f'))
                elif k == 'speedX':
                    cx = 'X'
                    if self.d[k] < 0: cx = 'R'
                    strout = '%6.1f %s' % (self.d[k], bargraph(self.d[k], -30, 300, 50, cx))
                elif k == 'speedY':  # This gets reversed for display to make sense.
                    strout = '%6.1f %s' % (self.d[k], bargraph(self.d[k] * -1, -25, 25, 50, 'Y'))
                elif k == 'speedZ':
                    strout = '%6.1f %s' % (self.d[k], bargraph(self.d[k], -13, 13, 50, 'Z'))
                elif k == 'z':
                    strout = '%6.3f %s' % (self.d[k], bargraph(self.d[k], .3, .5, 50, 'z'))
                elif k == 'trackPos':  # This gets reversed for display to make sense.
                    cx = '<'
                    if self.d[k] < 0: cx = '>'
                    strout = '%6.3f %s' % (self.d[k], bargraph(self.d[k] * -1, -1, 1, 50, cx))
                elif k == 'stucktimer':
                    if self.d[k]:
                        strout = '%3d %s' % (self.d[k], bargraph(self.d[k], 0, 300, 50, "'"))
                    else:
                        strout = 'Not stuck!'
                elif k == 'rpm':
                    g = self.d['gear']
                    if g < 0:
                        g = 'R'
                    else:
                        g = '%1d' % g
                    strout = bargraph(self.d[k], 0, 10000, 50, g)
                elif k == 'angle':
                    asyms = [
                        "  !  ", ".|'  ", "./'  ", "_.-  ", ".--  ", "..-  ",
                        "---  ", ".__  ", "-._  ", "'-.  ", "'\.  ", "'|.  ",
                        "  |  ", "  .|'", "  ./'", "  .-'", "  _.-", "  __.",
                        "  ---", "  --.", "  -._", "  -..", "  '\.", "  '|."]
                    rad = self.d[k]
                    deg = int(rad * 180 / PI)
                    symno = int(.5 + (rad + PI) / (PI / 12))
                    symno = symno % (len(asyms) - 1)
                    strout = '%5.2f %3d (%s)' % (rad, deg, asyms[symno])
                elif k == 'skid':  # A sensible interpretation of wheel spin.
                    frontwheelradpersec = self.d['wheelSpinVel'][0]
                    skid = 0
                    if frontwheelradpersec:
                        skid = .5555555555 * self.d['speedX'] / frontwheelradpersec - .66124
                    strout = bargraph(skid, -.05, .4, 50, '*')
                elif k == 'slip':  # A sensible interpretation of wheel spin.
                    frontwheelradpersec = self.d['wheelSpinVel'][0]
                    slip = 0
                    if frontwheelradpersec:
                        slip = ((self.d['wheelSpinVel'][2] + self.d['wheelSpinVel'][3]) -
                                (self.d['wheelSpinVel'][0] + self.d['wheelSpinVel'][1]))
                    strout = bargraph(slip, -5, 150, 50, '@')
                else:
                    strout = str(self.d[k])
            out += "%s: %s\n" % (k, strout)
        return out


class DriverAction():
    # What the driver is intending to do (i.e. send to the server).
    # Composes something like this for the server:
    # (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus 0)(meta 0) or
    # (accel 1)(brake 0)(gear 1)(steer 0)(clutch 0)(focus -90 -45 0 45 90)(meta 0)

    def __init__(self):
        self.actionstr = str()
        # "d" is for data dictionary.
        self.d = {'accel': 0.2,
                  'brake': 0,
                  'clutch': 0,
                  'gear': 1,
                  'steer': 0,
                  'focus': [-90, -45, 0, 45, 90],
                  'meta': 0
                  }

    def clip_to_limits(self):
        """There pretty much is never a reason to send the server
        something like (steer 9483.323). This comes up all the time
        and it's probably just more sensible to always clip it than to
        worry about when to. The "clip" command is still a snakeoil
        utility function, but it should be used only for non standard
        things or non obvious limits (limit the steering to the left,
        for example). For normal limits, simply don't worry about it."""
        self.d['steer'] = clip(self.d['steer'], -1, 1)
        self.d['brake'] = clip(self.d['brake'], 0, 1)
        self.d['accel'] = clip(self.d['accel'], 0, 1)
        self.d['clutch'] = clip(self.d['clutch'], 0, 1)
        if self.d['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.d['gear'] = 0
        if self.d['meta'] not in [0, 1]:
            self.d['meta'] = 0
        if type(self.d['focus']) is not list or min(self.d['focus']) < -180 or max(self.d['focus']) > 180:
            self.d['focus'] = 0

    def __repr__(self):
        self.clip_to_limits()
        out = str()
        for k in self.d:
            out += '(' + k + ' '
            v = self.d[k]
            if not type(v) is list:
                out += '%.3f' % v
            else:
                out += ' '.join([str(x) for x in v])
            out += ')'
        return out
        return out + '\n'

    def fancyout(self):
        # Specialty output for useful monitoring of bot's effectors.
        out = str()
        od = self.d.copy()
        od.pop('gear', '')  # Not interesting.
        od.pop('meta', '')  # Not interesting.
        od.pop('focus', '')  # Not interesting. Yet.
        for k in sorted(od):
            if k == 'clutch' or k == 'brake' or k == 'accel':
                strout = ''
                strout = '%6.3f %s' % (od[k], bargraph(od[k], 0, 1, 50, k[0].upper()))
            elif k == 'steer':  # Reverse the graph to make sense.
                strout = '%6.3f %s' % (od[k], bargraph(od[k] * -1, -1, 1, 50, 'S'))
            else:
                strout = str(od[k])
            out += "%s: %s\n" % (k, strout)
        return out


# == Misc Utility Functions
def destringify(s):
    '''makes a string into a value or a list of strings into a list of
    values (if possible)'''
    if not s: return s
    if type(s) is str:
        try:
            return float(s)
        except ValueError:
            print("Could not find a value in %s" % s)
            return s
    elif type(s) is list:
        if len(s) < 2:
            return destringify(s[0])
        else:
            return [destringify(i) for i in s]


def drive_example(c):
    S, R = c.S.d, c.R.d

    v_brake = 20
    v_max = 18
    threshold = 0.05
    speed_car = S['speedX']
    angle_car = S['angle']
    acc = 0
    brake = 0
    steer = 0

    if speed_car > v_brake:
        brake = 1
    if speed_car < v_max:
        acc = 1
    if angle_car > threshold:
        steer = 1
    if angle_car < -threshold:
        steer = -1

    R['accel'] = acc
    R['brake'] = brake
    R['steer'] = steer

    return


# ================ MAIN ================
if __name__ == "__main__":
    C = Client(p=3001)  # 3100 per torcs 3.1
    for step in range(C.maxSteps, 0, -1):
        C.get_servers_input()
        drive_example(C)
        C.respond_to_server()
    C.shutdown()
