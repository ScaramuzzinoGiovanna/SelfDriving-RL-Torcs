class Discr():
    def __init__(self):
        self.speed = [0, 10, 20, 40]
        self.angle = [-0.3, -0.22, -0.15, -0.07, 0, 0.07, 0.15, 0.22, 0.3]
        self.dist = [-1, -0.7, -0.4, -0.2, 0, 0.2, 0.4, 0.7, 1]

    def discr_state(self, state):
        trackPos = state['trackPos']
        v_car = state['speedX']
        angle = state['angle']
        ang = self.discr_angle(angle)
        s = self.discr_speed(v_car)
        d = self.discr_distance(trackPos)

        return [s, ang, d]

    def discr_angle(self, angle):
        if angle <= -0.3:
            a = -0.3
        if angle <= -0.2 and angle > - 0.3:
            a = -0.22
        if angle <= -0.1 and angle > - 0.2:
            a = -0.15
        if angle < -0.05 and angle > - 0.1:
            a = -0.07
        elif angle >= -0.05 and angle <= 0.05:
            a = 0
        elif angle > 0.05 and angle < 0.1:
            a = 0.07
        elif angle >= 0.1 and angle < 0.2:
            a = 0.15
        elif angle >= 0.2 and angle < 0.3:
            a = 0.22
        elif angle >= 0.3:
            a = 0.3
        return a

    def discr_speed(self, speed):
        if speed < 10:
            s = 0
        if speed >= 10 and speed < 18:
            s = 10
        elif speed >= 18 and speed <= 20:
            s = 20
        elif speed > 20:
            s = 40
        return s

    def discr_distance(self, trackPos):
        if trackPos <= -0.7:
            t = -1
        elif trackPos <= -0.5 and trackPos > -0.7:
            t = -0.7
        elif trackPos <= -0.3 and trackPos > -0.5:
            t = -0.4
        elif trackPos < -0.1 and trackPos > -0.3:
            t = -0.2
        elif trackPos >= -0.1 and trackPos <= 0.1:
            t = 0
        elif trackPos > 0.1 and trackPos < 0.3:
            t = 0.2
        elif trackPos >= 0.3 and trackPos < 0.5:
            t = 0.4
        elif trackPos >= 0.5 and trackPos < 0.7:
            t = 0.7
        if trackPos >= 0.7:
            t = 1
        return t
