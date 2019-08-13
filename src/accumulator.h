#ifndef ACCUMULATOR_T
#define ACCUMULATOR_T

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

#include <mpi.h>

#include <Cabana_AoSoA.hpp>
#include <Cabana_Distributor.hpp>
#include <Cabana_ExecutionPolicy.hpp>
#include <Cabana_Parallel.hpp>
#include <Cabana_Slice.hpp>

#include "types.h"
#include "grid.h"
#include "fields.h"

/*
class accumulator_t {
    public:
      float jx[4];   // jx0@(0,-1,-1),jx1@(0,1,-1),jx2@(0,-1,1),jx3@(0,1,1)
      float jy[4];   // jy0@(-1,0,-1),jy1@(-1,0,1),jy2@(1,0,-1),jy3@(1,0,1)
      float jz[4];   // jz0@(-1,-1,0),jz1@(1,-1,0),jz2@(-1,1,0),jz3@(1,1,0)

      accumulator_t() :
          jx { 0.0f, 0.0f, 0.0f, 0.f },
          jy { 0.0f, 0.0f, 0.0f, 0.f },
          jz { 0.0f, 0.0f, 0.0f, 0.f }
      {
          // empty
      }
};
*/

void clear_accumulator_array(
        field_array_t& fields,
        accumulator_array_t& accumulators,
        size_t nx, // TODO: we can probably pull these out of global params..
        size_t ny,
        size_t nz
);

void unload_accumulator_array(
        field_array_t& fields,
        accumulator_array_t& accumulators,
        size_t nx, // TODO: we can probably pull these out of global params..
        size_t ny,
        size_t nz,
        size_t ng,
        real_t dx,
        real_t dy,
        real_t dz,
        real_t dt
);

// TODO: find a better file for this
class ghosts_t {
// NOTE: this is an inefficient way to pass particles around
//  1. a function to pass kokkos::views would be useful
//  2. being able to specify what index each item recived
//      should be placed would be very useful

  public:    
    ghosts_t(
        const size_t nx,
        const size_t ny,
        const size_t nz,
        const size_t num_ghosts,
        const size_t num_real_cells,
        const size_t num_cells,
        const int dims[3],
        MPI_Comm mpi_comm
    );

    void collapse_boundaries( accumulator_array_t accumulators );

    void scatter( accumulator_array_t accumulators );

    void scatter( field_array_t fields );

  private: 
    Kokkos::View<int*, MemorySpace> _exports;
    Kokkos::View<int*, MemorySpace> _ghost_sends;
    Kokkos::View<int*, MemorySpace> _ghost_recvs;
    accumulator_aosoa_t _accumulator_buffer;
    field_array_t _field_buffer;
    std::shared_ptr<Cabana::Distributor<MemorySpace>> _distributor;
    MPI_Comm _mpi_comm;

};        

#endif // header guard
