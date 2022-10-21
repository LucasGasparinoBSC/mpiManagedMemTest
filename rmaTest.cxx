#include <mpi.h>
#ifdef _OPENACC
#include <openacc.h>
#endif
#include <nvToolsExt.h>
#include <iostream>
#include <cstdint>

int main(int argc, char const *argv[])
{
    // Initialize MPI
    MPI_Init(NULL, NULL);

    // Get the number of processes and set each rank
    int numProcs;
    int myRank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

#ifdef _OPENACC
    // Get the number of devices
    int numDevices;
    acc_init(acc_device_nvidia);
    numDevices = acc_get_num_devices(acc_device_nvidia);

    // Bind a rank to a device
    int deviceNum = myRank % numDevices;
    acc_set_device_num(deviceNum, acc_device_nvidia);
    std::cout << "Rank " << myRank << " is bound to device " << deviceNum << std::endl;
#endif

    // Each rank creates 3 arrays of size 1000
    int arrSize = 512*512*512;
    std::cout << "Rank " << myRank << " is allocating " << 3.0*arrSize*sizeof(double)/1.0e9 << " GBs" << std::endl;
    double *arr1 = new double[arrSize];
    double *arr2 = new double[arrSize];
    double *arr3 = new double[arrSize];

    // Each rank initializes its arrays
    nvtxRangePushA("Initialize arrays");
#ifdef _OPENACC
    #pragma acc parallel loop
#endif
    for (int i = 0; i < arrSize; i++)
    {
        arr1[i] = 1.0f;
        arr2[i] = 2.0f;
        arr3[i] = 0.0f;
    }
    nvtxRangePop();

    // For each rank, perform a saxpy
    nvtxRangePushA("Perform saxpy");
#ifdef _OPENACC
    #pragma acc parallel loop
#endif
    for (int i = 0; i < arrSize; i++)
    {
        arr3[i] = arr1[i] * 2.0f + arr2[i];
    }
    nvtxRangePop();

    // Each rank performs a partial sum of arr3
    nvtxRangePushA("Perform partial sum");
    double partialSum = 0.0f;
#ifdef _OPENACC
    #pragma acc parallel loop reduction(+:partialSum)
#endif
    for (int i = 0; i < arrSize; i++)
    {
        partialSum += arr3[i];
    }
    nvtxRangePop();

    // Create a window for the partial sums
    nvtxRangePushA("Create window");
    double totalSum = 0.0f;
    MPI_Win win;
    MPI_Win_create(&totalSum, sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    nvtxRangePop();

    // Each rank performs an RMA operation to sum the partial sums
    nvtxRangePushA("Perform RMA operation");
    MPI_Win_fence(0, win);
    for (int irank = 0; irank < numProcs; irank++)
    {
        MPI_Accumulate(&partialSum, 1, MPI_DOUBLE, irank, 0, 1, MPI_DOUBLE, MPI_SUM, win);
    }
    MPI_Win_fence(0, win);
    nvtxRangePop();

    // Print the total sum from each processsor
    std::cout << "Rank " << myRank << " has a total sum of " << totalSum << std::endl;

    // Check results
    double expectedSum = 4.0f * arrSize * numProcs;
    if (totalSum == expectedSum)
    {
        std::cout << "Rank " << myRank << " has the correct result" << std::endl;
    }
    else
    {
        std::cerr << "Rank " << myRank << " has the incorrect result" << std::endl;
        std::cerr << "Expected: " << expectedSum << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Free memory
    delete[] arr1;
    delete[] arr2;
    delete[] arr3;

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
