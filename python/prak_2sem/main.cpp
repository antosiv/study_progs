#include <iostream>
#include <list>
#include <vector>
#include <utility>

struct Node {
    size_t step[4] = {0, 0, 0, 0};
    std::list<Node *> sons;

    Node() = default;

    explicit Node(const size_t init_step[4]) {
        for (size_t i = 0; i < 4; ++i) step[i] = init_step[i];
    }

    ~Node() {
        for (auto elem : sons) delete elem;
    }
};

class StepCalculator {
private:
    std::vector<std::pair<size_t, size_t >> &coord_player;
    std::vector<std::pair<size_t, size_t >> &coord_opponent;
    std::vector<std::vector<int>> &figures_map;
    const size_t depth;

    std::pair<std::tuple<int, int, int, int>, int> optimal_step_rec (std::pair<size_t, size_t > fig) {

    }
public:
    StepCalculator(
            std::vector<std::pair<size_t, size_t >> &coord_player,
            std::vector<std::pair<size_t, size_t >> &coord_opponent,
            std::vector<std::vector<int>> &figures_map,
            size_t depth
    ) : coord_opponent(coord_opponent), coord_player(coord_player), figures_map(figures_map), depth(depth) {}

    std::tuple<int, int, int, int> optimal_step () {

    }
};

int main(int argc, char *argv[]) {

    if (argc != 19)
        return 0;

    std::vector<std::vector<int>> figures_map;

    for (size_t i = 0; i < 8; ++i) {
        figures_map.emplace_back(std::vector<int>());
        figures_map[i].reserve(8);
        for (size_t j = 0; j < 8; ++j)
            figures_map[i].push_back(std::atoi(argv[i * 8 + j]));

    }

    const int uid = std::atoi(argv[16]);
    const auto depth = static_cast<size_t>(std::atoi(argv[17]));

    std::vector<std::pair<size_t, size_t >> coord_player;
    coord_player.reserve(8);
    std::vector<std::pair<size_t, size_t >> coord_opponent;
    coord_opponent.reserve(8);

    for (size_t i = 0; i < 8; ++i) {
        for (size_t j = 0; j < 8; ++j) {
            if (uid * figures_map[i][j] > 0) coord_player.emplace_back(std::make_pair(i, j));
            else if (uid * figures_map[i][j] < 0) coord_opponent.emplace_back(std::make_pair(i, j));
        }
    }

    Node player_root, opponent_root;
    for (auto checker : coord_player) {
        
    }
    return 0;
}