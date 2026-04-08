class Agent:
    def __init__(self, a,b,c,d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def chooseAction(self, observation):
        w,x,y,z = observation

        calculation = self.a*w + self.b*x + self.c*y + self.d*z

        if calculation > 0:
            return 1
        else:
            return 0
