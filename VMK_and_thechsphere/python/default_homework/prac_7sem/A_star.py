import numpy as np
import pygame
from time import sleep


class AStar:
    def __init__(self, rows, cols, screen_width=640, screen_height=480):
        # render stuff
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.cell_width = screen_width // cols
        self.cell_height = screen_height // rows
        self.screen = pygame.display.set_mode((screen_width, screen_height))

        # logic stuff
        self.field = np.random.sample((rows, cols))
        self.start = (np.random.randint(rows), np.random.randint(cols))
        self.fin = (np.random.randint(rows), np.random.randint(cols))
        self.heuristic_map = np.ones((rows, cols)) * rows * cols
        self.heuristic_map[self.start[0], self.start[1]] = 0
        self.border = [self.start]
        self.closed_cells = np.zeros((rows, cols))

    def _dist_metric(self, cell):
        return (
                abs(self.fin[0] - cell[0]) / self.field.shape[0] +
                abs(self.fin[1] - cell[1]) / self.field.shape[1]
        ) * 15

    def _step(self):
        cell = self.border[0]
        for variant in (cell[0] - 1, cell[1]),\
                       (cell[0] + 1, cell[1]),\
                       (cell[0], cell[1] + 1),\
                       (cell[0], cell[1] - 1):
            if variant[0] < 0 or\
                    variant[0] == self.field.shape[0] or\
                    variant[1] < 0 or\
                    variant[1] == self.field.shape[1] or\
                    self.closed_cells[variant[0], variant[1]] == 1:
                continue
            if self.heuristic_map[variant[0], variant[1]] > \
                    self.field[variant[0], variant[1]] + self.heuristic_map[cell[0], cell[1]]:
                self.heuristic_map[variant[0], variant[1]] = \
                    self.field[variant[0], variant[1]] + self.heuristic_map[cell[0], cell[1]]
                if variant not in self.border:
                    self.border.append(variant)
        self.border.remove(cell)
        self.closed_cells[cell[0], cell[1]] = 1
        self.border.sort(
            key=lambda x: self.heuristic_map[x[0], x[1]] + self._dist_metric(x)
        )

    def _render_step(self):
        for i in range(self.field.shape[0]):
            for j in range(self.field.shape[1]):
                if (i, j) == self.start:
                    color = pygame.Color('green')
                elif (i, j) == self.fin:
                    color = pygame.Color('red')
                elif self.closed_cells[i, j] == 1:
                    color = pygame.Color('white')
                elif (i, j) in self.border:
                    color = pygame.Color('grey')
                else:
                    intensity = hex(int(255 * self.field[i, j]))[2:]
                    if len(intensity) == 1:
                        intensity = '0' + intensity
                    color = \
                        pygame.Color('#0000' + intensity.upper() + '00')

                pygame.draw.rect(
                    self.screen,
                    color,
                    (j * self.cell_width, i * self.cell_height, self.cell_width, self.cell_height)
                )

    def _draw_way(self):
        way = []
        curr_cell = self.fin
        while curr_cell != self.start:
            min_var = None
            for variant in (curr_cell[0] - 1, curr_cell[1]),\
                           (curr_cell[0] + 1, curr_cell[1]),\
                           (curr_cell[0], curr_cell[1] + 1),\
                           (curr_cell[0], curr_cell[1] - 1):
                if variant[0] < 0 or \
                        variant[0] == self.field.shape[0] or \
                        variant[1] < 0 or \
                        variant[1] == self.field.shape[1]:
                    continue
                if min_var is None or \
                        self.heuristic_map[variant[0], variant[1]] < self.heuristic_map[min_var[0], min_var[1]]:
                    min_var = variant
            way.append(min_var)
            curr_cell = min_var
        for cell in way:
            pygame.draw.rect(
                self.screen,
                pygame.Color('green'),
                (cell[1] * self.cell_width, cell[0] * self.cell_height, self.cell_width, self.cell_height)
            )

    def run(self):
        pygame.init()
        pygame.display.set_caption('A*')
        self._render_step()
        pygame.display.flip()

        running = True
        working = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if not working:
                sleep(0.1)
                continue
            if self.closed_cells[self.fin[0], self.fin[1]] == 1 or len(self.border) == 0:
                self._draw_way()
                pygame.display.flip()
                working = False
                sleep(0.1)
                continue
            self._step()
            self._render_step()
            pygame.display.flip()
            sleep(0.1)
        pygame.quit()


if __name__ == '__main__':
    a_star = AStar(50, 50)
    a_star.run()
