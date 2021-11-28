#pragma once

#include <vector>
#include <iostream>
#include <cmath>
#define DEBUG
// #define RESULT_DEBUG
template <typename Container>
void printVector(const Container &cont)
{
    for (auto const &x : cont)
    {
        std::cout << x << " ";
    }
    std::cout << '\n';
}
namespace hdist
{

    enum class Algorithm : int
    {
        Jacobi = 0,
        Sor = 1
    };

    struct State
    {
        int room_size = 10;
        float block_size = 2;
        int source_x = room_size / 2;
        int source_y = room_size / 2;
        float source_temp = 100;
        float border_temp = 36;
        float tolerance = 0.02;
        float sor_constant = 4.0;
        Algorithm algo = hdist::Algorithm::Jacobi;

        bool operator==(const State &that) const = default;
    };

    struct Alt
    {
    };

    constexpr static inline Alt alt{};

    struct Grid
    {
        std::vector<double> data0, data1;
        size_t current_buffer = 0;
        size_t length;

        explicit Grid(size_t size,
                      double border_temp,
                      double source_temp,
                      size_t x,
                      size_t y)
            : data0(size * size), data1(size * size), length(size)
        {
            for (size_t i = 0; i < length; ++i)
            {
                for (size_t j = 0; j < length; ++j)
                {
                    if (i == 0 || j == 0 || i == length - 1 || j == length - 1)
                    {
                        this->operator[]({i, j}) = border_temp;
                    }
                    else if (i == x && j == y)
                    {
                        this->operator[]({i, j}) = source_temp;
                    }
                    else
                    {
                        this->operator[]({i, j}) = 0;
                    }
                }
            }
        }

        std::vector<double> &get_current_buffer()
        {
            if (current_buffer == 0)
                return data0;
            return data1;
        }

        double &operator[](std::pair<size_t, size_t> index)
        {
            return get_current_buffer()[index.first * length + index.second];
        }

        double &operator[](std::tuple<Alt, size_t, size_t> index)
        {
            return current_buffer == 1 ? data0[std::get<1>(index) * length + std::get<2>(index)] : data1[std::get<1>(index) * length + std::get<2>(index)];
        }

        void switch_buffer()
        {
            current_buffer = !current_buffer;
        }
    };

    struct UpdateResult
    {
        bool stable;
        double temp;
    };

    UpdateResult update_single(size_t i, size_t j, Grid &grid, const State &state)
    {
        UpdateResult result{};
        if (i == 0 || j == 0 || i == state.room_size - 1 || j == state.room_size - 1)
        {
            result.temp = state.border_temp;
        }
        else if (i == state.source_x && j == state.source_y)
        {
            result.temp = state.source_temp;
        }
        else
        {
            auto sum = (grid[{i + 1, j}] + grid[{i - 1, j}] + grid[{i, j + 1}] + grid[{i, j - 1}]);
            switch (state.algo)
            {
            case Algorithm::Jacobi:
                result.temp = 0.25 * sum;
                break;
            case Algorithm::Sor:
                result.temp = grid[{i, j}] + (1.0 / state.sor_constant) * (sum - 4.0 * grid[{i, j}]);
                break;
            }
        }
        result.stable = std::fabs(grid[{i, j}] - result.temp) < state.tolerance;
        return result;
    }

    bool calculate(const State &state, Grid &grid)
    {
        bool stabilized = true;

        switch (state.algo)
        {
        case Algorithm::Jacobi:
            for (size_t i = 0; i < state.room_size; ++i)
            {
                for (size_t j = 0; j < state.room_size; ++j)
                {
                    auto result = update_single(i, j, grid, state);
                    stabilized &= result.stable;
                    grid[{alt, i, j}] = result.temp;
                }
            }
            grid.switch_buffer();
            break;
        case Algorithm::Sor:
            for (auto k : {0, 1})
            {
                for (size_t i = 0; i < state.room_size; i++)
                {
                    for (size_t j = 0; j < state.room_size; j++)
                    {
                        if (k == ((i + j) & 1))
                        {
                            auto result = update_single(i, j, grid, state);
                            stabilized &= result.stable;
                            grid[{alt, i, j}] = result.temp;
                        }
                        else
                        {
                            grid[{alt, i, j}] = grid[{i, j}];
                        }
                    }
                }
                grid.switch_buffer();
            }
        }
        return stabilized;
    };
    // MPI usage ---
    struct MPI_Grid_Part
    {
        std::vector<double> data0, data1;
        int current_buffer = 0;
        int room_size, rows, comm_size, rank;

        explicit MPI_Grid_Part(int rows,
                               int room_size,
                               double border_temp,
                               double source_temp,
                               int x,
                               int y,
                               int rank, int comm_size)
            : data0((rows + 2) * room_size), data1((rows + 2) * room_size), room_size(room_size), rows(rows + 2), comm_size(comm_size), rank(rank)
        {
            // add two ghost rows;
            for (size_t i = 0; i < rows + 2; ++i)
            {
                for (size_t j = 0; j < room_size; ++j)
                {
                    if (j == 0 || j == room_size - 1 || (rank == 0 && (i == 1)) || (rank == comm_size - 1 && (i == rows + 1)))
                    {
                        this->operator[]({i, j}) = border_temp;
                    }

                    int row_offset = (rank < room_size % comm_size) ? rank * (room_size / comm_size + 1 + 2) : (room_size % comm_size) + rank * (room_size / comm_size + 2);
                    if ((i - 1) >= 0 && (i - 1) < rows && (i - 1 + row_offset == x && j == y))
                    {
                        this->operator[]({i, j}) = source_temp;
                    }
                    else
                    {
                        this->operator[]({i, j}) = 0;
                    }
                }
            }
        }

        std::vector<double> &get_current_buffer()
        {
            if (current_buffer == 0)
                return data0;
            return data1;
        }

        double &operator[](std::pair<size_t, size_t> index)
        {
            return get_current_buffer()[index.first * room_size + index.second];
        }

        double &operator[](std::tuple<Alt, size_t, size_t> index)
        {
            return current_buffer == 1 ? data0[std::get<1>(index) * room_size + std::get<2>(index)] : data1[std::get<1>(index) * room_size + std::get<2>(index)];
        }

        void switch_buffer()
        {
            current_buffer = !current_buffer;
        }
    };

    UpdateResult MPI_update_single(size_t i, size_t j, MPI_Grid_Part &grid, const State &state)
    {
        UpdateResult result{};
        int row_offset = (grid.rank < grid.room_size % grid.comm_size) ? grid.rank * (grid.room_size / grid.comm_size + 1) : (grid.room_size % grid.comm_size) + grid.rank * (grid.room_size / grid.comm_size);

        if (j == 0 || j == state.room_size - 1 || (grid.rank == 0 && (i == 1)) || (grid.rank == grid.comm_size - 1 && (i == grid.rows - 2)))
        {
            result.temp = state.border_temp;
        }
        else if (i - 1 == state.source_x - row_offset && j == state.source_y)
        {
            result.temp = state.source_temp;
        }
        else
        {
            auto sum = (grid[{i + 1, j}] + grid[{i - 1, j}] + grid[{i, j + 1}] + grid[{i, j - 1}]);
            result.temp = 0.25 * sum;
        }
        result.stable = std::fabs(grid[{i, j}] - result.temp) < state.tolerance;
        return result;
    }

    bool calculate(const State &state, MPI_Grid_Part &grid)
    {
        bool stabilized = true;
#ifdef DEBUG
        printf("------ rank: %d ; rows: %d; room_size: %d----- \n", grid.rank, grid.rows, grid.room_size);
#endif // DEBUG
        for (size_t i = 1; i < grid.rows - 1; ++i)
        {
            for (size_t j = 0; j < grid.room_size; ++j)
            {
                auto result = MPI_update_single(i, j, grid, state);
                stabilized &= result.stable;
                grid[{alt, i, j}] = result.temp;
#ifdef DEBUG
                printf("%f ", grid[{alt, i, j}]);
#endif // DEBUG
            }
#ifdef DEBUG
            printf("\n");
#endif // DEBUG
        }
#ifdef DEBUG
        printf("------ rank: %d ----- \n", grid.rank);
#endif // DEBUG

        grid.switch_buffer();
        return stabilized;
    }

} // namespace hdist