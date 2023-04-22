class Direction:
    CENTER = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    SHIFT = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]
    all = [UP, RIGHT, DOWN, LEFT]
    logs = ['CENTER', 'UP', 'RIGHT', 'DOWN', 'LEFT']

    @classmethod
    def shift(cls, pos, direction):
        return pos[0] + cls.SHIFT[direction][0], pos[1] + cls.SHIFT[direction][1]


class Resource:
    ICE = 0
    ORE = 1
    WATER = 2
    METAL = 3
    POWER = 4

