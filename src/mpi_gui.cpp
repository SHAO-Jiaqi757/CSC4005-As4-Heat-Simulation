#include <graphic/graphic.hpp>
#include <imgui_impl_sdl.h>
#include <cstring>
#include <chrono>
#include <hdist/hdist.hpp>
#include <mpi.h>

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}
void get_slice(int &start_row, int &end_row, int rank, int comm_size, int room_size);
ImColor temp_to_color(double temp)
{
    auto value = static_cast<uint8_t>(temp / 100.0 * 255.0);
    return {value, 0, 255 - value};
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    // UNUSED(argc, argv);
    int rank, comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    bool finished = false, local_finished = false;
    bool first = true;
    int recv_buff_size, room_size, x, y;
    double border_temp, source_temp;
    std::vector<double> result;
    std::vector<int> sendcnts, dipls;
    static hdist::State current_state;
    static std::chrono::high_resolution_clock::time_point begin, end; // in rank 0
    int total_iter;
    if (rank == 0)
    {
        if (argc < 3)
        {
            printf("Error: Useage: mpi <room_size> <iteration>\n");
            return 0;
        }
        room_size = atoi(argv[1]);
        total_iter = atoi(argv[2]);
        x = room_size / 2;
        y = room_size / 2;
        border_temp = current_state.border_temp;
        source_temp = current_state.source_temp;
        result.resize(room_size * room_size);

        begin = std::chrono::high_resolution_clock::now();
        int avg_rows = room_size / comm_size;
        int remain_rows = room_size % comm_size;
        int offset_row = 0;
        for (int rank_id = 0; rank_id < comm_size; rank_id++)
        {
            int local_rows = (rank_id < remain_rows) ? avg_rows + 1 : avg_rows;
            sendcnts.push_back(local_rows * room_size);
            dipls.push_back(offset_row * room_size);
            offset_row += local_rows;
        }
    }
    MPI_Bcast(&room_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&y, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&border_temp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&source_temp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    current_state.room_size = room_size;
    current_state.source_x = x;
    current_state.source_y = y;
    int rows = rank < room_size % comm_size ? room_size / comm_size + 1 : room_size / comm_size;
    auto grid = hdist::MPI_Grid_Part{
        rows, room_size, border_temp, source_temp, x, y, rank, comm_size};

    int first_ghost_row_offset = 0;                                          // get the first ghost row;
    int first_row_offset = 1 * grid.room_size;                               // send the first row;
    int last_ghost_row_offset = (grid.rows_with_ghost - 1) * grid.room_size; // get the last ghost row;
    int last_row_offset = (grid.rows_with_ghost - 2) * grid.room_size;       // send the last row;

    if (rank == 0)
    {
        int tmp_iter = 0;
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
        ImGui::Text("Room Size: %d", room_size);
        ImGui::Text("Block Size: %f", current_state.block_size);
        ImGui::Text("Source Temp: %f", current_state.source_temp);
        ImGui::Text("Border Temp: %f", current_state.border_temp);
        ImGui::Text("Source X: %d", current_state.source_x);
        ImGui::Text("Source Y: %d", current_state.source_y);
        ImGui::Text("Tolerance: %f", current_state.tolerance);
        ImGui::Text("Max Iteration: %d", total_iter);
        ImGui::Text("Algorithm: Jacobi");

        if (!finished) {
            // int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            //                  int dest, int sendtag, void *recvbuf, int recvcount,
            //                  MPI_Datatype recvtype, int source, int recvtag,
            //                  MPI_Comm comm, MPI_Status *status)
            MPI_Sendrecv(grid.get_current_buffer().data() + last_row_offset, grid.room_size, MPI_DOUBLE,
                         rank + 1, 1, grid.get_current_buffer().data() + last_ghost_row_offset, grid.room_size,
                         MPI_DOUBLE, rank + 1, 1,
                         MPI_COMM_WORLD, &status);
            // calculation...
            local_finished = calculate(current_state, grid);
            MPI_Gatherv(grid.get_current_buffer().data() + first_row_offset, rows * room_size, MPI_DOUBLE, result.data(), sendcnts.data(), dipls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_finished, &finished, 1, MPI_BYTE, MPI_LAND, 0, MPI_COMM_WORLD);
            MPI_Bcast(&finished, 1, MPI_BYTE, 0, MPI_COMM_WORLD);
            tmp_iter++;
            if (tmp_iter >= total_iter) finished = true;
            if (finished)
                end = std::chrono::high_resolution_clock::now();
        } else {
            ImGui::Text("Stable or Reach max iteration %d for %lld ns", total_iter, std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
        }

        const ImVec2 p = ImGui::GetCursorScreenPos();
        float x = p.x + current_state.block_size, y = p.y + current_state.block_size;
        for (size_t i = 0; i < current_state.room_size; ++i) {
            for (size_t j = 0; j < current_state.room_size; ++j) {
                // auto temp = grid[{i, j}];
                auto temp = result[i*current_state.room_size + j];
                auto color = temp_to_color(temp);
                draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + current_state.block_size, y + current_state.block_size), color);
                y += current_state.block_size;
            }
            x += current_state.block_size;
            y = p.y + current_state.block_size;
        }
        
        ImGui::End(); });
    }
    else if (rank != 0)
    {
        while (1)
        {
            MPI_Sendrecv(grid.get_current_buffer().data() + first_row_offset, grid.room_size, MPI_DOUBLE,
                         rank - 1, 1, grid.get_current_buffer().data() + first_ghost_row_offset, grid.room_size,
                         MPI_DOUBLE, rank - 1, 1,
                         MPI_COMM_WORLD, &status);
            if (rank != comm_size - 1)
            {
                MPI_Sendrecv(grid.get_current_buffer().data() + last_row_offset, grid.room_size, MPI_DOUBLE,
                             rank + 1, 1, grid.get_current_buffer().data() + last_ghost_row_offset, grid.room_size,
                             MPI_DOUBLE, rank + 1, 1,
                             MPI_COMM_WORLD, &status);
            }

            // calculation...
            local_finished = calculate(current_state, grid);

            // gather;
            MPI_Gatherv(grid.get_current_buffer().data() + first_row_offset, rows * room_size, MPI_DOUBLE, result.data(), sendcnts.data(), dipls.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            // send finished signal;
            MPI_Reduce(&local_finished, &finished, 1, MPI_BYTE, MPI_LAND, 0, MPI_COMM_WORLD);
            MPI_Bcast(&finished, 1, MPI_BYTE, 0, MPI_COMM_WORLD);
            // if (finished)
            //     break;
        }
    }
    MPI_Finalize();
}
void get_slice(int &start_row, int &end_row, int rank, int comm_size, int room_size)
{
    int avg = room_size / comm_size;
    int rem = room_size % comm_size;
    start_row = (rank < rem) ? (avg + 1) * rank : rem + avg * rank;
    end_row = (rank < rem) ? start_row + avg + 1 : start_row + avg;
}