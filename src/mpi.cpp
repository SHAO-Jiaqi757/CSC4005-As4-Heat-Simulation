#include <cstring>
#include <chrono>
#include <hdist/hdist.hpp>
#include <mpi.h>

template <typename... Args>
void UNUSED(Args &&...args [[maybe_unused]]) {}
void get_slice(int &start_row, int &end_row, int rank, int comm_size, int room_size);

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
    static std::chrono::high_resolution_clock::time_point begin, end; // in rank 0

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

        begin = std::chrono::high_resolution_clock::now();
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
        std::vector<double> result(room_size * room_size);

        int offset = rows;
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
            bool part_finish = calculate(current_state, grid);
            memcpy(result.data(), grid.get_current_buffer().data() + first_row_offset, rows * room_size * sizeof(double));
            for (int rank_id = 1; rank_id < comm_size; ++rank_id)
            {

                // gather
                int rows_part = (rank_id < (room_size % comm_size)) ? room_size / comm_size + 1 : room_size / comm_size;
                MPI_Recv(result.data() + offset * room_size, rows_part * room_size, MPI_DOUBLE, rank_id, 0, MPI_COMM_WORLD, &status);
                offset += rows_part;

                // TODO: MPI_Reduce
                bool recv_finish;

                MPI_Recv(&recv_finish, 1, MPI_BYTE, rank_id, 9, MPI_COMM_WORLD, &status);
#ifdef MYDEBUG
                printf("I am rank %d, receive stable signal %d from rank %d \n", rank, recv_finish, rank_id);

#endif

                part_finish &= recv_finish;
                MPI_Send(&part_finish, 1, MPI_BYTE, rank_id, 8, MPI_COMM_WORLD);
#ifdef MYDEBUG
                printf("I am rank %d, sending finished signal %d to rank %d \n", rank, part_finish, rank_id);

#endif
                finished = part_finish;
            }
            offset = rows;
        }
        end = std::chrono::high_resolution_clock::now();
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
#ifdef MYDEBUG
            printf("===== rank %d send to rank %d, receive from rank %d ==== \n", rank, rank - 1, rank - 1);
            // printVector(grid.get_current_buffer());
            printf(" ----------------------------------------------------------------\n");
#endif
            if (rank != comm_size - 1)
            {

                // int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                //                  int dest, int sendtag, void *recvbuf, int recvcount,
                //                  MPI_Datatype recvtype, int source, int recvtag,
                //                  MPI_Comm comm, MPI_Status *status)
                MPI_Sendrecv(grid.get_current_buffer().data() + last_row_offset, grid.room_size, MPI_DOUBLE,
                             rank + 1, 1, grid.get_current_buffer().data() + last_ghost_row_offset, grid.room_size,
                             MPI_DOUBLE, rank + 1, 1,
                             MPI_COMM_WORLD, &status);
#ifdef MYDEBUG
                printf("===== rank %d send to rank %d, receive from rank %d ====== \n", rank, rank + 1, rank + 1);
                // printVector(grid.get_current_buffer());
                printf(" ----------------------------------------------------------------\n");
#endif
            }

            // calculation...
            bool part_finish = calculate(current_state, grid);

            // gather;
            MPI_Send(grid.get_current_buffer().data() + first_row_offset, rows * room_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

            // send finished signal;
            MPI_Send(&part_finish, 1, MPI_BYTE, 0, 9, MPI_COMM_WORLD);
#ifdef MYDEBUG
            printf("I am rank %d, sending stable signal %d to rank 0 \n", rank, part_finish);

#endif
            MPI_Recv(&finished, 1, MPI_BYTE, 0, 8, MPI_COMM_WORLD, &status);
#ifdef MYDEBUG
            printf("I am rank %d, receive finished signal %d from rank 0 \n", rank, finished);
#endif
            if (finished)
                break;
        }
    }
    MPI_Finalize();
    return 0;
}
