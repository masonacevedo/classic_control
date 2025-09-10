def make_decision(observation):
    cart_pos, cart_vel, pole_angle, pole_angular_vel = observation

    if pole_angle > 0:
        return 1
    else:
        return 0