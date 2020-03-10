import numpy as np
import pygame
from time import sleep


class TrafficJam:
    def __init__(
            self,
            acceleration_distance_const,
            deceleration_distance_const,
            acceleration,
            deceleration,
            road_length,
            screen_width=640,
            screen_height=480,
            car_length=10
    ):
        # render stuff
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.cell_width = car_length
        self.cell_height = screen_height // (road_length // (screen_width // car_length))
        self.n_cells_in_row = screen_width // car_length
        self.screen = pygame.display.set_mode((screen_width, screen_height))

        # logic stuff
        self.acceleration_distance_const = acceleration_distance_const
        self.deceleration_distance_const = deceleration_distance_const
        self.acceleration = acceleration
        self.deceleration = deceleration
        self.road = np.zeros(road_length, dtype=int)

    def _step(self, born_car=False):
        updated_road = np.zeros(self.road.size, dtype=int)
        if born_car:
            self.road[0] = np.random.randint(low=1, high=10)
        for pos, speed in enumerate(self.road):
            if speed == 0:
                continue
            updated_road[pos] = speed
            next_car = pos + 1
            while next_car < self.road.size and self.road[next_car] == 0:
                next_car += 1
            if next_car != self.road.size and (next_car - pos) < self.deceleration_distance_const * speed:
                updated_road[pos] -= self.deceleration
                updated_road[pos] = max(0, updated_road[pos])
            elif next_car == self.road.size or (next_car - pos) > self.acceleration_distance_const * speed:
                updated_road[pos] += self.acceleration

            if pos + updated_road[pos] > updated_road.size:
                updated_road[pos] = 0
                continue
            else:
                updated_speed = updated_road[pos]
                updated_road[pos] = 0
                if pos + updated_speed < self.road.size:
                    if updated_road[pos + updated_speed] != 0:
                        taken = list()
                        taken.append(pos + updated_speed)

                        tmp_pos = pos + updated_speed - 1
                        while tmp_pos >= 0 and updated_road[tmp_pos] != 0:
                            taken.append(tmp_pos)
                            tmp_pos -= 1
                        tmp_pos = pos + updated_speed + 1
                        while tmp_pos < self.road.size and updated_road[tmp_pos] != 0:
                            taken.append(tmp_pos)
                            tmp_pos += 1
                        taken.sort()

                        tmp_pos = min(taken) - 1
                        while len(taken) > 0 and tmp_pos >= 0:
                            updated_road[tmp_pos] = updated_road[taken[-1]]
                            updated_road[taken[-1]] = 0
                            taken.pop()
                    updated_road[pos + updated_speed] = updated_speed
        self.road = updated_road

    def _render_step(self, stopped_car=None):
        field = np.concatenate(
            [self.road, np.ones(self.n_cells_in_row - self.road.size % self.n_cells_in_row) * -1]
        ).reshape(-1, self.n_cells_in_row)
        if stopped_car is not None:
            field[stopped_car // field.shape[1], stopped_car % field.shape[1]] = -2
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                if field[i, j] == 0:
                    color = pygame.Color('white')
                elif field[i, j] == -1:
                    color = pygame.Color('black')
                elif field[i, j] == -2:
                    color = pygame.Color('yellow')
                else:
                    intensity = hex(min(255, int(255 * field[i, j] / 20)))[2:]
                    if len(intensity) == 1:
                        intensity = '0' + intensity
                    color = pygame.Color('#00' + intensity.upper() + '0000')

                pygame.draw.rect(
                    self.screen,
                    color,
                    (j * self.cell_width, i * self.cell_height, self.cell_width, self.cell_height)
                )

    def run(self):
        pygame.init()
        pygame.display.set_caption('Traffic jam')

        time_to_next = 0
        running = True
        while running:
            if time_to_next == 0:
                self._step(born_car=True)
                time_to_next = np.random.poisson()
            else:
                time_to_next -= 1
                self._step()
            god_stopped_car = None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == 32:
                        candidates = np.where(self.road != 0)[0]
                        if candidates.size != 0:
                            god_stopped_car = candidates[np.random.randint(low=0, high=candidates.size)]
                            self.road[god_stopped_car] = 1
            self._render_step(stopped_car=god_stopped_car)
            pygame.display.flip()
            sleep(0.3)
        pygame.quit()


if __name__ == '__main__':
    traffic_jam = TrafficJam(10, 2, 2, 1, 200)
    traffic_jam.run()
