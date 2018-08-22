import pygame
from pygame.locals import *
import numpy as np
import subprocess


class GameError(Exception):
    pass


class Checkers:
    def __init__(self, width=640, height=480, speed=10, g_type='ai_vs_ai', max_depth=2):
        self.width = width
        self.height = height
        self.type = g_type
        self.screen_size = width, height
        self.max_depth = max_depth

        self.screen = pygame.display.set_mode(self.screen_size)

        self.cell_width = self.width // 8
        self.cell_height = self.height // 8

        self.figures_map = None
        # Скорость протекания игры, для игры ai only
        self.speed = speed

    def run_g(self):
        if self.type == 'ai_vs_ai':
            self.run_ava()
        else:
            raise GameError()

    def run_ava(self):
        pygame.init()
        clock = pygame.time.Clock()
        pygame.display.set_caption('Checkers')

        self._draw_board()
        self.init_field()
        self._draw_figures()
        pygame.display.flip()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
            # pygame.display.flip()
            clock.tick(self.speed)
        pygame.quit()

    def _draw_board(self):
        colors = [pygame.Color('black'), pygame.Color('white')]
        for i in range(8):
            for j in range(8):
                pygame.draw.rect(self.screen, colors[(i + j) % 2],
                                 (j * self.cell_width, i * self.cell_height, self.cell_width, self.cell_height)
                                 )

    def _draw_figures(self):
        if self.figures_map is None:
            raise GameError()
        for i in range(8):
            for j in range(8):
                f_color = None
                if self.figures_map[i, j] == 1:
                    f_color = pygame.Color('red')
                elif self.figures_map[i, j] == -1:
                    f_color = pygame.Color('grey')
                elif self.figures_map[i, j] == 2:
                    f_color = pygame.Color('pink')
                elif self.figures_map[i, j] == -2:
                    f_color = pygame.Color('white')
                if f_color is not None:
                    pygame.draw.circle(self.screen,
                                       f_color,
                                       (
                                           j * self.cell_width + self.cell_width // 2,
                                           i * self.cell_height + self.cell_height // 2
                                        ),
                                       min(self.cell_width, self.cell_height) // 2
                                       )

    def init_field(self):
        self.figures_map = np.zeros((8, 8), dtype=np.int32)

        mask_1 = (np.arange(8) % 2) == 0
        mask_2 = (np.arange(8) % 2) == 1

        self.figures_map[0, mask_1] = 1
        self.figures_map[1, mask_2] = 1
        self.figures_map[6, mask_1] = -1
        self.figures_map[7, mask_2] = -1

    def get_opt_step(self, uid):
        params = ['./s_calc.out']
        for i in range(8):
            for j in range(8):
                params.append(str(self.figures_map[i, j]))

        params.append(str(uid))
        params.append(str(self.max_depth))

        return map(int, subprocess.run(params, stdout=subprocess.PIPE).stdout.split())


if __name__ == '__main__':
    a = Checkers()
    a.run_g()
