def gather(consts, t):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)
