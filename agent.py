class Agent:
    def __init__(self, params):
        if len(params) != 4:
            raise Exception("Invalid argument to agent construtor! Must be a list of 4 parameters.")
        self.params = params

    def chooseAction(self, observation):
        w,x,y,z = observation
        a,b,c,d = self.params

        calculation = a*w + b*x + c*y + d*z

        if calculation > 0:
            return 1
        else:
            return 0

    def __repr__(self):
        return str([self.a, self.b, self.c, self.d])