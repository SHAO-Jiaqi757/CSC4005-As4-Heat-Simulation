#include <cstring>
#include <chrono>
#include <hdist/hdist.hpp>
// #include "pthread_barrier.h"

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}
void get_slice(int &start_row, int &end_row, int thread_id, int thread_number);
int iter = 0;
int thread_number = 2;
void *pthread_calculate(void *arg);
bool pthread_routine();
// pthread_barrier_t barrier;
std::vector<bool> stable_vector(thread_number, 1);

static hdist::State current_state;
auto grid = hdist::Grid{
    static_cast<size_t>(current_state.room_size),
    current_state.border_temp,
    current_state.source_temp,
    static_cast<size_t>(current_state.source_x),
    static_cast<size_t>(current_state.source_y)};

int main(int argc, char **argv)
{
    // UNUSED(argc, argv);
    if (argc < 4)
    {
        printf("Error: Useage: pthread_gui <thread_number> <room_size> <iteration>\n");
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
        bool finished = pthread_routine();
        if (finished)
            break;
    }
    end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    printf("pthread_number: %d \n", thread_number);
    printf("problem_size: %d \n", current_state.room_size);
    printf("iterations: %d \n", iter);
    printf("duration(ns/iter): %lld \n", duration / iter);
}

bool pthread_routine()
{
    bool stabilized = true;
    std::vector<pthread_t> threads(thread_number);
    std::vector<int> thread_ids(thread_number);
    // pthread_barrier_init(&barrier, NULL, thread_number);
    for (int i = 0; i < thread_number; i++)
    {
        thread_ids[i] = i;
        pthread_create(&threads[i], nullptr, pthread_calculate, (void *)&thread_ids[i]);
    }
    for (int i = 0; i < thread_number; i++)
    {
        pthread_join(threads[i], NULL);
        stabilized &= stable_vector[i];
    }
    grid.switch_buffer();
    return stabilized;
}
void *pthread_calculate(void *arg)
{
    int thread_id = *(int *)arg;
    int start_row, end_row;
    stable_vector[thread_id] = 1;

    get_slice(start_row, end_row, thread_id, thread_number);
    // printf("thread [%d] >> start row: %d, end row: %d\n", thread_id, start_row, end_row);

    for (size_t i = start_row; i < end_row; ++i)
    {
        for (size_t j = 0; j < current_state.room_size; ++j)
        {
            auto result = update_single(i, j, grid, current_state);
            stable_vector[thread_id] = stable_vector[thread_id] & result.stable;
            grid[{hdist::alt, i, j}] = result.temp;
        }
    }
    return nullptr;
};

void get_slice(int &start_row, int &end_row, int thread_id, int thread_number)
{
    int avg = current_state.room_size / thread_number;
    int rem = current_state.room_size % thread_number;
    start_row = (thread_id < rem) ? (avg + 1) * thread_id : rem + avg * thread_id;
    end_row = (thread_id < rem) ? start_row + avg + 1 : start_row + avg;
}
