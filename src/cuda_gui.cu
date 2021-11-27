#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <chrono>
#include <hdist/hdist.cuh>
// #include <cuda_runtime.h>

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}

ImColor temp_to_color(double temp)
{
    auto value = static_cast<uint8_t>(temp / 100.0 * 255.0);
    return {value, 0, 255 - value};
}
__device__ __managed__ bool first = true;
__device__ __managed__ bool finished = false;

__device__ __managed__ static hdist::State *current_state = new hdist::State();
__device__ __managed__ static hdist::State *last_state = new hdist::State();
__device__ __managed__ auto *grid = new hdist::Grid{
    static_cast<size_t>(current_state->room_size),
    current_state->border_temp,
    current_state->source_temp,
    static_cast<size_t>(current_state->source_x),
    static_cast<size_t>(current_state->source_y)};
__device__ __managed__ int thread_number;
__device__ void get_slice(int &start_row, int &end_row, int thread_id, int thread_number);
__device__ __managed__ bool stable_vector[current_state->room_size];

__global__ void cuda_thread_routine()
{
    bool stabilized = true;
    int thread_id = threadIdx.x;
    get_slice(start_row, start_row, thread_id, thread_number);

    for (size_t i = 0; i < current_state->room_size; ++i)
    {
        stable_vector[i] = 1;
        for (size_t j = 0; j < current_state->room_size; ++j)
        {
            auto result = update_single(i, j, *grid, *current_state);
            stable_vector[i] = stable_vector[i] & result->stable;
            *grid[1, i, j] = result.temp;
        }
    }

    grid->switch_buffer();
    for (size_t i = 0; i < current_state->room_size; ++i)
    {
        stabilized &= stable_vector[i];
    }
    finished = stabilized;
}

int main(int argc, char **argv)
{
    // UNUSED(argc, argv);
    if (argc < 2)
    {
        printf("Error: No <thread_number> found! \nUseage: cuda_gui <thread_number> \n");
        return 0;
    }
    thread_number = atoi(argv[1]);
    printf("thread_number: %d \n", thread_number);

    static std::chrono::high_resolution_clock::time_point begin, end;
    graphic::GraphicContext context{"Assignment 4"};
    context.run([&](graphic::GraphicContext *context [[maybe_unused]], SDL_Window *)
                {
        auto io = ImGui::GetIO();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("Assignment 4", nullptr,
                     ImGuiWindowFlags_NoMove
                     | ImGuiWindowFlags_NoCollapse
                     | ImGuiWindowFlags_NoTitleBar
                     | ImGuiWindowFlags_NoResize);
        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);
        ImGui::DragInt("Room Size", &current_state->room_size, 10, 200, 1600, "%d");
        ImGui::DragFloat("Block Size", &current_state->block_size, 0.01, 0.1, 10, "%f");
        ImGui::DragFloat("Source Temp", &current_state->source_temp, 0.1, 0, 100, "%f");
        ImGui::DragFloat("Border Temp", &current_state->border_temp, 0.1, 0, 100, "%f");
        ImGui::DragInt("Source X", &current_state->source_x, 1, 1, current_state->room_size - 2, "%d");
        ImGui::DragInt("Source Y", &current_state->source_y, 1, 1, current_state->room_size - 2, "%d");
        ImGui::DragFloat("Tolerance", &current_state->tolerance, 0.01, 0.01, 1, "%f");
        ImGui::Text("Algorithm: Jacobi");

        if (current_state->room_size != last_state->room_size) {
            grid = new hdist::Grid{
                    static_cast<size_t>(current_state->room_size),
                    current_state->border_temp,
                    current_state->source_temp,
                    static_cast<size_t>(current_state->source_x),
                    static_cast<size_t>(current_state->source_y)};
            first = true;
        }

        if (*current_state != *last_state) {
            last_state = current_state;
            finished = false;
        }

        if (first) {
            first = false;
            finished = false;
            begin = std::chrono::high_resolution_clock::now();
        }

        if (!finished) {
            // finished = hdist::calculate(current_state, grid);
            cuda_thread_routine<<<1, thread_number>>>();

            if (finished) end = std::chrono::high_resolution_clock::now();
        } else {
            ImGui::Text("stabilized in %lld ns", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
        }

        // drawing...;
        const ImVec2 p = ImGui::GetCursorScreenPos();
        float x = p.x + current_state->block_size, y = p.y + current_state->block_size;
        for (size_t i = 0; i < current_state->room_size; ++i) {
            for (size_t j = 0; j < current_state->room_size; ++j) {
                auto temp = grid[i, j];
                auto color = temp_to_color(temp);
                draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + current_state->block_size, y + current_state->block_size), color);
                y += current_state->block_size;
            }
            x += current_state->block_size;
            y = p.y + current_state->block_size;
        }
        ImGui::End(); });
    delete grid;
    delete current_state;
    delete last_state;
}
__device__ void get_slice(int &start_row, int &end_row, int thread_id, int thread_number)
{
    int avg = current_state.room_size / thread_number;
    int rem = current_state.room_size % thread_number;
    start_row = (thread_id < rem) ? (avg + 1) * thread_id : rem + avg * thread_id;
    end_row = (thread_id < rem) ? start_row + avg + 1 : start_row + avg;
}
