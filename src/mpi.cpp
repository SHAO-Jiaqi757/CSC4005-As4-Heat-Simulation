#include <cstring>
#include <chrono>
#include <hdist/hdist.hpp>
#include <mpi.h>

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}
void get_slice(int &start_row, int &end_row, int rank, int comm_size, int room_size);
void sequential(hdist::State current_state, int total_iter); // when using 1 cores;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    // UNUSED(argc, argv);
    int rank, comm_size, total_iter;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    std::vector<double> result;
    bool finished = false, local_finished = false;
    int room_size, x, y;
    double border_temp, source_temp;

    static hdist::State current_state;
    static std::chrono::high_resolution_clock::time_point begin, end; // in rank 0

    std::vector<int> sendcnts, dipls;
    if (rank == 0)
    {
        if (argc < 3)
        {
            printf("Error: Useage: mpi <room_size> <iteration>\n");
            return 0;
        }
        current_state.room_size = atoi(argv[1]);
        total_iter = atoi(argv[2]);

        room_size = current_state.room_size;
        x = current_state.source_x;
        y = current_state.source_y;
        border_temp = current_state.border_temp;
        source_temp = current_state.source_temp;
        result.resize(room_size * room_size);
        begin = std::chrono::high_resolution_clock::now();
        if (comm_size == 1)
        {
            sequential(current_state, total_iter);
            return 0;
        }
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
        // int offset = rows;
        while (!finished)
        {
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
            // MPI_Reduce(&local_finished, &finished, 1, MPI_BYTE, MPI_LAND, 0, MPI_COMM_WORLD);
            if (tmp_iter >= total_iter)
                finished = true;

            MPI_Bcast(&finished, 1, MPI_BYTE, 0, MPI_COMM_WORLD);
            tmp_iter++;
            // offset = rows;
        }
        end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        printf("cores: %d \n", comm_size);
        printf("problem_size: %d \n", current_state.room_size);
        printf("iterations: %d \n", total_iter);
        printf("duration(ns/iter): %lld \n", duration / total_iter);
#ifdef RESULT_DEBUG
        printf("result (length: %d): ", result.size());
        printVector(result);
#endif
    }
    else if (rank != 0)
    {
        while (!finished)
        {
            // int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
            //                  int dest, int sendtag, void *recvbuf, int recvcount,
            //                  MPI_Datatype recvtype, int source, int recvtag,
            //                  MPI_Comm comm, MPI_Status *status)
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

            // MPI_Reduce(&local_finished, &finished, 1, MPI_BYTE, MPI_LAND, 0, MPI_COMM_WORLD);
            MPI_Bcast(&finished, 1, MPI_BYTE, 0, MPI_COMM_WORLD);
            if (finished)
                break;
        }
    }
    MPI_Finalize();
    return 0;
}

void sequential(hdist::State current_state, int total_iter)
{
    auto begin = std::chrono::high_resolution_clock::now();
    auto grid = hdist::Grid{
        static_cast<size_t>(current_state.room_size),
        current_state.border_temp,
        current_state.source_temp,
        static_cast<size_t>(current_state.source_x),
        static_cast<size_t>(current_state.source_y)};
    for (int i = 0; i < total_iter; i++)
    {
        calculate(current_state, grid);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    printf("cores: %d \n", 1);
    printf("problem_size: %d \n", current_state.room_size);
    printf("iterations: %d \n", total_iter);
    printf("duration(ns/iter): %lld \n", duration / total_iter);
    MPI_Finalize();
}