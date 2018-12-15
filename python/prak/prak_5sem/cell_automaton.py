import pygame
from pygame.locals import *
import random
import math


class AutomatonError(Exception):
    pass


class Automaton:
    def __init__(self, width=640, height=480, cell_size=10, speed=10,
                 automaton_radius=3, max_state=150, plus_const=10,
                 randomize=False, random_density=0.5
                 ):
        self.width = width
        self.height = height
        self.cell_size = cell_size

        # Устанавливаем размер окна
        self.screen_size = width, height
        # Создание нового окна
        self.screen = pygame.display.set_mode(self.screen_size)

        # Вычисляем количество ячеек по вертикали и горизонтали
        self.cell_width = self.width // self.cell_size
        self.cell_height = self.height // self.cell_size

        # Объект в котором инкапсулирован внутренний механизм работы автомата
        self.insides = CellList(
            configuration=Mishaw(radius=automaton_radius, max_state=max_state,
                                 plus_const=plus_const
                                 ),
            nrow=self.cell_height, ncol=self.cell_width,
            randomize=randomize, max_state=max_state, random_density=random_density
        )

        # Скорость протекания игры
        self.speed = speed

        # Список цветов для отрисовки
        self.colors = []
        step = 360 // (max_state - 1)
        for i in range(max_state + 1):
            r = int(128 + math.cos(math.radians(i * step + 90)) * 127)
            g = int(128 + math.cos(math.radians(i * step + 330)) * 127)
            b = int(128 + math.cos(math.radians(i * step + 210)) * 127)
            curr_color = hex(0)[:2] + self._my_hex(r, 2) + self._my_hex(g, 2) + self._my_hex(b, 2)
            self.colors.append(pygame.Color(curr_color))

    @staticmethod
    def _my_hex(numb, size):
        res = ''
        if numb // (16 ** (size - 1)) == 0:
            while numb // (16 ** (size - 1)) == 0 and numb != 0:
                size -= 1
                res += '0'
        res += hex(numb)[2:]
        return res

    def _draw_grid(self):
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, pygame.Color('black'),
                             (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, pygame.Color('black'), (0, y), (self.width, y))

    def _draw_cell_list(self):
        for i in range(self.cell_height):
            for j in range(self.cell_width):
                if self.insides[i][j] != 0:
                    pygame.draw.rect(self.screen, self.colors[self.insides[i][j]],
                                     (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                                     )
                else:
                    pygame.draw.rect(self.screen, pygame.Color('white'),
                                     (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                                     )
        self.insides.update()

    def run(self):
        pygame.init()
        clock = pygame.time.Clock()
        pygame.display.set_caption('Chemical Reaction')
        self.screen.fill(pygame.Color('white'))
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
            self._draw_grid()
            self._draw_cell_list()
            pygame.display.flip()
            clock.tick(self.speed)
        pygame.quit()


class CellList:
    def __init__(self, configuration, nrow=100, ncol=100, randomize=False, max_state=1, random_density=0.5):
        if random_density > 1:
            raise AssertionError
        self.data = []
        self.new_data = []
        self.size = (nrow, ncol)
        if randomize:
            for i in range(nrow):
                self.data.append([])
                for j in range(ncol):
                    value = random.randint(0, max_state * 10000)
                    value -= value % (int(10000*(1 - random_density)) * max_state)
                    if value > 0:
                        value = random.randint(1, max_state)
                    self.data[i].append(value)
        else:
            self.data = [[0 for _ in range(ncol)] for _ in range(nrow)]
            self.data[nrow // 2][ncol // 2] = 1
        self.new_data = [[0 for _ in range(ncol)] for _ in range(nrow)]
        self.configuration = configuration

    def update(self):
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                self.new_data[i][j] = self.configuration.new_state(self._get_neighbours(i, j))
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                self.data[i][j] = self.new_data[i][j]

    def _get_neighbours(self, row, col):
        if row > self.size[0] or col > self.size[1]:
            raise AssertionError
        res = []
        for j in range(self.configuration.radius):
            for i in range(self.configuration.radius):
                res.append(self.data[(row - self.configuration.radius // 2 + j) % self.size[0]]
                           [(col - self.configuration.radius // 2 + i) % self.size[1]]
                           )
        return res

    def __getitem__(self, item):
        return self.data[item]


class Mishaw:

    def __init__(self, max_state=3, radius=3, plus_const=10):
        self.radius = radius
        self.max_state = max_state
        self.plus_const = plus_const

    def new_state(self, pre_state):
        state_sum = 0
        for i in range(self.radius ** 2):
            if i != (self.radius ** 2) // 2:
                state_sum += pre_state[i]
        if pre_state[(self.radius ** 2) // 2] == 0:
            if state_sum < 5:
                return 0
            elif state_sum < 100:
                return 2
            else:
                return 3
        elif pre_state[self.radius ** 2 // 2] == self.max_state:
            return 0
        else:
            res = state_sum // self.radius ** 2 + self.plus_const
            if res > self.max_state:
                res = self.max_state
            return res


class SimpleAutomatonConfiguration:
    @staticmethod
    def _to_bin(size, numb):
        if numb >= 2 ** size:
            raise AutomatonError
        res = ''
        for i in reversed(range(size)):
            res += str(numb // (2 ** i))
            numb %= 2 ** i
        return res

    def __init__(self, radius=3, automaton_type=22):
        self.radius = radius
        self.conf = {}
        bin_type = self._to_bin(2 ** radius, automaton_type)
        for i in range(2 ** radius):
            self.conf.setdefault(self._to_bin(radius, i), int(bin_type[2 ** radius - i - 1]))

    def __getitem__(self, item):
        return self.conf[item]


if __name__ == '__main__':
    game = Automaton(width=1000, height=600, cell_size=5,
                     automaton_radius=3, max_state=150, plus_const=20,
                     randomize=True, random_density=0.0005
                     )
    game.run()
