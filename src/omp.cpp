#include <cstring>
#include <chrono>
#include <hdist/hdist.hpp>
template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}

int iter = 0;
bool first = true;
bool finished = false;
static hdist::State current_state;
auto grid = hdist::Grid{
    static_cast<size_t>(current_state.room_size),
    current_state.border_temp,
    current_state.source_temp,
    static_cast<size_t>(current_state.source_x),
    static_cast<size_t>(current_state.source_y)};
int thread_number;
void omp_thread_routine();
// std::vector<bool> stable_vector(current_state.room_size, 1);

int main(int argc, char **argv)
{
    // UNUSED(argc, argv);
    if (argc < 4)
    {
        printf("Error: Useage: omp_gui <thread_number> <room_size> <iteration>\n");
        return 0;
    }
    thread_number = atoi(argv[1]);
    current_state.room_size = atoi(argv[2]);
    iter = atoi(argv[3]);

    static std::chrono::high_resolution_clock::time_point begin, end;

    grid = hdist::Grid{
        static_cast<size_t>(current_state.room_size),
        current_state.border_temp,
        current_state.source_temp,
        static_cast<size_t>(current_state.source_x),
        static_cast<size_t>(current_state.source_y)};

    begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; i++)
    {
        omp_thread_routine();
    }
    end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    printf("openmp_thread: %d \n", thread_number);
    printf("problem_size: %d \n", current_state.room_size);
    printf("iterations: %d \n", iter);
    printf("duration(ns/iter): %lld \n", duration / iter);
}

void omp_thread_routine()
{
#pragma omp parallel for num_threads(thread_number) shared(grid, current_state)
    for (size_t i = 0; i < current_state.room_size; ++i)
    {
        for (size_t j = 0; j < current_state.room_size; ++j)
        {
            auto result = update_single(i, j, grid, current_state);
            grid[{hdist::alt, i, j}] = result.temp;
        }
    }
#pragma omp barrier
    grid.switch_buffer();
}