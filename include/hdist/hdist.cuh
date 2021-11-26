#pragma once
// #include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#define __device__
#define __managed__
class Managed
{
public:
    __host__ void *operator new(size_t len)
    {
        void *ptr;
        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();
        return ptr;
    }

    __host__ void operator delete(void *ptr)
    {
        cudaDeviceSynchronize();
        cudaFree(ptr);
    }
};

namespace hdist
{

    struct State : public Managed
    {
        int room_size = 300;
        float block_size = 2;
        int source_x = room_size / 2;
        int source_y = room_size / 2;
        float source_temp = 100;
        float border_temp = 36;
        float tolerance = 0.02;
        float sor_constant = 4.0;

        bool operator==(const State &that) const = default;
    };

    struct Grid : public Managed
    {
        const max_size = 1024;
        double data0[max_size * max_size], data1[max_size * max_size];
        size_t current_buffer = 0;
        size_t length;

        explicit Grid(size_t size,
                      double border_temp,
                      double source_temp,
                      size_t x,
                      size_t y)
            : length(size)
        {
            for (size_t i = 0; i < length; ++i)
            {
                for (size_t j = 0; j < length; ++j)
                {
                    if (i == 0 || j == 0 || i == length - 1 || j == length - 1)
                    {
                        this->operator[](i, j) = border_temp;
                    }
                    else if (i == x && j == y)
                    {
                        this->operator[](i, j) = source_temp;
                    }
                    else
                    {
                        this->operator[](i, j) = 0;
                    }
                }
            }
        }

        __device__ double[] & get_current_buffer()
        {
            if (current_buffer == 0)
                return data0;
            return data1;
        }

        __device__ double &operator[](size_t i, size_t j)
        {
            return get_current_buffer()[i * length + j];
        }

        __device__ double &operator[](size_t flag, size_t i, size_t j)
        {
            return current_buffer == 1 ? data0[i * length + j(index)] : data1[i * length + j];
        }

        __device__ void switch_buffer()
        {
            current_buffer = !current_buffer;
        }
    };

    struct UpdateResult : public Managed
    {
        bool stable;
        double temp;
    };

    __device__ *UpdateResult update_single(size_t i, size_t j, Grid &grid, const State &state)
    {
        UpdateResult *result = new UpdateResult();
        if (i == 0 || j == 0 || i == state.room_size - 1 || j == state.room_size - 1)
        {
            result->temp = state.border_temp;
        }
        else if (i == state.source_x && j == state.source_y)
        {
            result->temp = state.source_temp;
        }
        else
        {
            auto sum = (grid[i + 1, j] + grid[i - 1, j] + grid[i, j + 1] + grid[i, j - 1]);
            result.temp = 0.25 * sum;
        }
        double fabs = (grid[i, j] - result->temp) > 0 ? grid[i, j] - result->temp : result->temp - grid[i, j];
        result->stable = fabs < state.tolerance;
        return result;
    }
} // namespace hdist