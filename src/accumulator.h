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

class accumulator_ghosts_t {

  public:    
    void initialize( 
            size_t nx,
            size_t ny,
            size_t nz,
            size_t num_ghosts,
            size_t num_real_cells,
            size_t num_cells
    );

    void scatter( accumulator_array_t accumulators );

  private: 
    Kokkos::View<int*, MemorySpace> _acc_exports;
    Kokkos::View<int*, MemorySpace> _ghost_sends;
    Kokkos::View<int*, MemorySpace> _ghost_recvs;
    accumulator_aosoa_t _accumulator_buffer;
    std::shared_ptr<Cabana::Distributor<MemorySpace>>
        _accumulator_distributor;

};        

#endif // header guard
