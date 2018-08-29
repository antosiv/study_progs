import pygame
from pygame.locals import *
import numpy as np
import subprocess
import random
from time import sleep

class GameError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class Checkers:
    def __init__(self, width=640, height=480, speed=1, g_type='ai_vs_ai', recursion_depth=2):
        self.width = width
        self.height = height
        self.type = g_type
        self.screen_size = width, height
        self.recursion_depth = recursion_depth

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
            raise GameError('Unknown game type')

    def _draw_board(self):
        colors = [pygame.Color('black'), pygame.Color('white')]
        for i in range(8):
            for j in range(8):
                pygame.draw.rect(self.screen, colors[(i + j) % 2],
                                 (j * self.cell_width, i * self.cell_height, self.cell_width, self.cell_height)
                                 )

    def _draw_figures(self):
        if self.figures_map is None:
            raise GameError('Game field is not inited')
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
                    pygame.draw.circle(
                        self.screen,
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

    @staticmethod
    def get_neighbors(pos):
        return [
            ((pos[0] - 1, pos[1] - 1), (pos[0] - 2, pos[1] - 2)),
            ((pos[0] + 1, pos[1] - 1), (pos[0] + 2, pos[1] - 2)),
            ((pos[0] - 1, pos[1] + 1), (pos[0] - 2, pos[1] + 2)),
            ((pos[0] + 1, pos[1] + 1), (pos[0] + 2, pos[1] + 2))
        ]

    @staticmethod
    def in_shape(pos, shape):
        return 0 <= pos[0] < shape[0] and 0 <= pos[1] < shape[1]

    def get_eat_chains(self, pos, fig_type, fig_map, root_pos=None):
        res = {}
        for neighbor in self.get_neighbors(pos):
            if self.in_shape(neighbor[0], fig_map.shape) and\
                    self.in_shape(neighbor[1], fig_map.shape) and \
                    fig_map[neighbor[0]] == -1 * fig_type and \
                    fig_map[neighbor[1]] == 0 and\
                    (root_pos is None or root_pos[0] != neighbor[1][0] or root_pos[1] != neighbor[1][1]):
                    res[neighbor[1]] = self.get_eat_chains(neighbor[1], fig_type, fig_map, root_pos=pos)
        return res

    def get_chains_list(self, root_list, all_list, root_dict):
        if len(root_dict) == 0:
            return
        else:
            keys = list(root_dict.keys())
            for key in keys[1:]:
                all_list.append(root_list.copy())
                all_list[-1].append(key)
                self.get_chains_list(all_list[-1], all_list, root_dict[key])

            root_list.append(keys[0])
            self.get_chains_list(root_list, all_list, root_dict[keys[0]])

    def get_possible_steps(self, fig_type, fig_map):
        steps = []
        eats = []
        for figure in np.argwhere(fig_map == fig_type):

            # steps
            for pos in (figure[0] + fig_type, figure[1] + 1), (figure[0] + fig_type, figure[1] - 1):
                if self.in_shape(pos, fig_map.shape):
                    if fig_map[pos] == 0:
                        steps.append((tuple(figure), pos))
            # eats
            chains = self.get_eat_chains(figure, fig_type, fig_map)
            if len(chains) > 0:
                eats.append([])
                self.get_chains_list(eats[-1], eats, {tuple(figure): chains})

        return steps, eats

    def make_step(self, source_map, step):
        res_map = source_map.copy()
        if len(step) == 2 and max([abs(step[0][0] - step[1][0]), abs(step[0][1] - step[1][1])]) == 1:
            res_map[step[1]] = res_map[step[0]]
            res_map[step[0]] = 0
        else:
            for i in range(1, len(step)):
                res_map[step[i - 1][0] + (step[i][0] - step[i - 1][0]) // 2, step[i - 1][1] + (step[i][1] - step[i - 1][1]) // 2] = 0
                res_map[step[i]] = res_map[step[i - 1]]
                res_map[step[i - 1]] = 0
        return res_map

    def get_step_tree(self, root_dict, fig_type, recursion_depth, fig_map, alpha, beta):
        if recursion_depth == 0 or alpha > beta:
            return
        steps = self.get_possible_steps(fig_type, fig_map)
        for step in [*steps[0], *steps[1]]:
            t_step = tuple(step)
            root_dict[t_step] = dict()
            root_dict[t_step]['map'] = self.make_step(fig_map, step)
            root_dict[t_step]['score'] = root_dict[t_step]['map'].sum()
            alpha = max(alpha, root_dict[t_step]['score'])
            beta = min(beta, root_dict[t_step]['score'])
            root_dict[t_step]['sons'] = dict()
            self.get_step_tree(root_dict[t_step]['sons'], -1 * fig_type, recursion_depth - 1, root_dict[t_step]['map'], alpha, beta)

    def get_score(self, regimen, root):
        sons_scores = [self.get_score(-1 * regimen, son) for son in root['sons'].values()]
        if len(sons_scores) == 0:
            return root['score']
        else:
            if regimen == 1:
                score = min(sons_scores)
            else:
                score = max(sons_scores)
            return score

    def get_optimal_step(self, fig_type):
        tree = dict()
        self.get_step_tree(tree, fig_type, self.recursion_depth, self.figures_map, -float('Inf'), float('Inf'))
        scores = {step: self.get_score(fig_type, elem) for step, elem in tree.items()}
        scores_list = list(scores.values())
        if len(scores_list) == 0:
            return
        if fig_type == 1:
            best_score = max(scores_list)
        else:
            best_score = min(scores_list)
        steps = list(filter(lambda x: True if scores[x] == best_score else False, scores.keys()))
        if len(steps) == 0:
            return
        else:
            ind = random.randint(0, len(steps) - 1)
            return steps[ind], tree[steps[ind]]['map']

    def run_ava(self):
        pygame.init()
        pygame.display.set_caption('Checkers')

        self._draw_board()
        self.init_field()
        self._draw_figures()
        pygame.display.flip()

        running = True
        figure_type = 1
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
            tmp = self.get_optimal_step(figure_type)
            if tmp is None:
                continue
            else:
                self.figures_map = tmp[1]
                self._draw_board()
                self._draw_figures()
                pygame.display.flip()
                figure_type *= -1
                sleep(self.speed)
        pygame.quit()

    def test(self):
        self.run_ava()


if __name__ == '__main__':
    a = Checkers()
    a.test()
