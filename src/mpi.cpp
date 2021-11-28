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

    bool finished = false;
    int recv_buff_size, room_size, x, y;
    double border_temp, source_temp;
    static hdist::State current_state, last_state;

    // std::vector<int> sendcnts, dipls;
    // std::vector<double> send_data;
    if (rank == 0)
    {
        bool first = true;
        room_size = current_state.room_size;
        x = current_state.source_x;
        y = current_state.source_y;
        border_temp = current_state.border_temp;
        source_temp = current_state.source_temp;

        static std::chrono::high_resolution_clock::time_point begin, end;
        /*
        auto grid = hdist::Grid{
            static_cast<size_t>(current_state.room_size),
            current_state.border_temp,
            current_state.source_temp,
            static_cast<size_t>(current_state.source_x),
            static_cast<size_t>(current_state.source_y)};
        send_data = grid.get_current_buffer();
        // Scatterv -- sendcnts & displs
        int avg_rows = room_size / comm_size;
        int remain_rows = room_size % comm_size;
        int displ = 0;
        for (int rank_id = 0; rank_id < comm_size; i++)
        {
            int rows = rank_id < remain_rows ? avg_rows + 1 : avg_rows;
            sendcnts.push_back(rows * room_size);
            displs.push_back(displ);
            displ += rows * room_size;
        }
        */
    }
    MPI_Bcast(&room_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&y, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&border_temp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&source_temp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /*
        recv_buff_size = rank < (room_size % comm_size) ?: (room_size / comm_size + 1) * room_size : (room_size / comm_size) * room_size;

        std::vector<double> recv_data(recv_buff_size);
        MPI_Scatterv(send_data.data(), sendcnts.data(), displs.data(), MPI_DOUBLE, recv_data.data(), recv_buff_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    */
    int rows = rank < room_size % comm_size ? room_size / comm_size + 1 : room_size / comm_size;
    auto grid = hdist::MPI_Grid_Part
    {
        rows, room_size, border_temp, source_temp, x, y, rank, comm_size
    }
    int recv_offset0 = 0;                                // get the first ghost row;
    int send_offset0 = 1 * grid.room_size;               // send the first row;
    int recv_offset1 = (grid.rows - 1) * grid.room_size; // get the last ghost row;
    int send_offset1 = (grid.rows - 2) * grid.room_size; // send the last row;

    if (rank == 0)
    {
        std::vector<double> result(room_size * room * size);
        int offset = rows * room_size;
        while (1)
        {
            MPI_Sendrecv(grid.get_current_buffer().data() + send_offset1, grid.room_size, MPI_DOUBLE, 1, 1, grid.get_current_buffer().data() + recv_offset1, grid.room_size, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &status);

            // calculation...
            bool part_finish = calculate(current_state, grid);
            memcpy(result.data(), gird.get_current_buffer().data() + send_offset1, rows * room_size);
            for (int rank_id = 1; rank_id < comm_size; ++rank_id)
            {

                // gather
                int rows_part = rank_id < room_size % comm_size ? room_size / comm_size + 1 : room_size / comm_size;
                MPI_Recv(result.data() + offset * room_size, rows_part * room_size, MPI_DOUBLE, rank_id, 0, MPI_COMM_WORLD, &status);
                offset += rows_part;

                // send finished signal
                bool recv_finish;
                MPI_Recv(&part_finish, 1, MPI_CHAR, rank_id, 9, MPI_COMM_WORLD, &status);
                part_finish &= recv_finish;
                MPI_Send(&part_finish, 1, MPI_CHAR, 0, 9, MPI_COMM_WORLD);

                finished = part_finish;
                if (finished)
                    break;
            }
        }
    }
    else if (rank != 0)
    {
        while (1)
        {
            if (rank != comm_size - 1)
            {
                MPI_Sendrecv(grid.get_current_buffer().data() + send_offset0, grid.room_size, MPI_DOUBLE, rank - 1, 1, grid.get_current_buffer().data() + recv_offset0, grid.room_size, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &status);
                MPI_Sendrecv(grid.get_current_buffer().data() + send_offset1, grid.room_size, MPI_DOUBLE, rank + 1, 1, grid.get_current_buffer().data() + recv_offset1, grid.room_size, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &status);
            }
            else
            { // the last row;
                MPI_Sendrecv(grid.get_current_buffer().data() + send_offset0, grid.room_size, MPI_DOUBLE, rank - 1, 1, grid.get_current_buffer().data() + recv_offset0, grid.room_size, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &status);
            }
            // calculation...
            bool part_finish = calculate(current_state, grid);

            // gather;
            MPI_Send(grid.get_curren_buffer().data() + send_offset1, rows * room_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            // send finished signal;
            MPI_Send(&part_finish, 1, MPI_CHAR, 0, 9, MPI_COMM_WORLD);
            MPI_Recv(&finished, 1, MPI_CHAR, 0, 9, MPI_COMM_WORLD);
            if (finished)
                break;
        }
    }
