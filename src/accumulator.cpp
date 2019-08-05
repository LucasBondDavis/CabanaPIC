// TODO: add namespace?

#include "accumulator.h"

void clear_accumulator_array(
        field_array_t& fields,
        accumulator_array_t& accumulators,
        size_t nx, // TODO: we can probably pull these out of global params..
        size_t ny,
        size_t nz
)
{
    auto _clean_accumulator = KOKKOS_LAMBDA(const int i)
    {
        /*
           a0(i,JX_OFFSET+0) = 0;
           a0(i+y_offset,JX_OFFSET+1) = 0;
           a0(i+z_offset,JX_OFFSET+2) = 0;
           a0(i+y_offset+z_offset,JX_OFFSET+3) = 0;

           a0(i,JY_OFFSET+0) = 0;
           a0(i+z_offset,JY_OFFSET+1) = 0;
           a0(i+y_offset,JY_OFFSET+2) = 0;
           a0(i+y_offset+z_offset,JY_OFFSET+3) = 0;

           a0(i,JZ_OFFSET+0) = 0;
           a0(i+x_offset,JZ_OFFSET+1) = 0;
           a0(i+y_offset,JZ_OFFSET+2) = 0;
           a0(i+x_offset+y_offset,JZ_OFFSET+3) = 0;
         */

      for (int j = 0; j < ACCUMULATOR_VAR_COUNT; j++)
      {
          for (int k = 0; k < ACCUMULATOR_VAR_COUNT; k++)
          {
              accumulators(i, j, k) = 0.0;
          }
      }
    };

    Kokkos::RangePolicy<ExecutionSpace> exec_policy( 0, fields.size() );
    Kokkos::parallel_for( exec_policy, _clean_accumulator, "clean_accumulator()" );
}


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
)
{

    auto jfx = fields.slice<FIELD_JFX>();
    auto jfy = fields.slice<FIELD_JFY>();
    auto jfz = fields.slice<FIELD_JFZ>();

    // TODO: give these real values
    real_t cx = 0.25 / (dy * dz * dt);
    real_t cy = 0.25 / (dz * dx * dt);
    real_t cz = 0.25 / (dx * dy * dt);

    // This is a hang over for VPIC's nasty type punting
    const size_t JX_OFFSET = 0;
    const size_t JY_OFFSET = 4;
    const size_t JZ_OFFSET = 8;

    size_t x_offset = 1; // VOXEL(x+1,y,  z,   nx,ny,nz);
    size_t y_offset = (1*nx); // VOXEL(x,  y+1,z,   nx,ny,nz);
    size_t z_offset = (1*nx*ny); // VOXEL(x,  y,  z+1, nx,ny,nz);

    // TODO: we have to be careful we don't reach past the ghosts here
    auto _unload_accumulator = KOKKOS_LAMBDA( const int x, const int y, const int z )
    {
        // Original:
        // f0->jfx += cx*( a0->jx[0] + ay->jx[1] + az->jx[2] + ayz->jx[3] );
        int i = VOXEL(x,y,z, nx,ny,nz,ng);

        jfx(i) = cx*(
                    accumulators(i,                   accumulator_var::jx, 0) +
                    accumulators(i+y_offset,          accumulator_var::jx, 1) +
                    accumulators(i+z_offset,          accumulator_var::jx, 2) +
                    accumulators(i+y_offset+z_offset, accumulator_var::jx, 3)
                );

        jfy(i) = cy*(
                    accumulators(i,                   accumulator_var::jy, 0) +
                    accumulators(i+z_offset,          accumulator_var::jy, 1) +
                    accumulators(i+y_offset,          accumulator_var::jy, 2) +
                    accumulators(i+y_offset+z_offset, accumulator_var::jy, 3)
                );

        jfz(i) = cz*(
                    accumulators(i,                   accumulator_var::jz, 0) +
                    accumulators(i+x_offset,          accumulator_var::jz, 1) +
                    accumulators(i+y_offset,          accumulator_var::jz, 2) +
                    accumulators(i+x_offset+y_offset, accumulator_var::jz, 3)
                );
    };

    Kokkos::MDRangePolicy< Kokkos::Rank<3> > non_ghost_policy( {ng,ng,ng}, {nx+ng, ny+ng, nz+ng} ); // Try not to into ghosts // TODO: dry this
    Kokkos::parallel_for( non_ghost_policy, _unload_accumulator, "unload_accumulator()" );

    /* // Crib sheet for old variable names
    a0  = &a(x,  y,  z  );
    ax  = &a(x-1,y,  z  );
    ay  = &a(x,  y-1,z  );
    az  = &a(x,  y,  z-1);
    ayz = &a(x,  y-1,z-1);
    azx = &a(x-1,y,  z-1);
    axy = &a(x-1,y-1,z  )
    */

}

// find the cell where ghosted parts go
int mirror( int cell, int nx, int ny, int nz, int num_ghosts ) {
    int ix, iy, iz;
    RANK_TO_INDEX( cell, ix, iy, iz, nx+2*num_ghosts, ny+2*num_ghosts );
    if ( ix < num_ghosts )
        ix += nx;
    else if ( ix > (nx-1) + num_ghosts )
        ix -= nx;
    //if ( iy < num_ghosts ) //ENABLE FOR 3d, breaks 1d
    //    iy += ny;
    //else if ( iy > (ny-1) + num_ghosts )
    //    iy -= ny;
    //if ( iz < num_ghosts )
    //    iz += nz;
    //else if ( iz > (nz-1) + num_ghosts )
    //    iz -= nz;
    return VOXEL( ix, iy, iz, nx, ny, nz, num_ghosts );
}

accumulator_ghosts_t::accumulator_ghosts_t( size_t nx, size_t ny, size_t nz,
                 size_t num_ghosts, size_t num_real_cells, size_t num_cells )
{
    
    // Get mpi info
    int comm_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    int comm_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    const int prev_rank = ( comm_rank == 0 ) ? comm_size - 1 : comm_rank - 1;
    const int next_rank = ( comm_rank == comm_size - 1 ) ? 0 : comm_rank + 1;

    // set constant for aosoa sizes
    const size_t num_ghost_cells = num_cells - num_real_cells;

    // create accumulator export ranks and ids
    Kokkos::View<int*,MemorySpace> acc_exports(
        Kokkos::ViewAllocateWithoutInitializing( "acc_exports" ),
        num_ghost_cells );
    Kokkos::View<int*, MemorySpace>::HostMirror acc_exports_host =
        Kokkos::create_mirror( acc_exports );

    Kokkos::View<int*, MemorySpace> ghost_sends( 
        Kokkos::ViewAllocateWithoutInitializing( "ghost_sends" ),
        num_ghost_cells );
    Kokkos::View<int*, MemorySpace>::HostMirror ghost_sends_host =
        Kokkos::create_mirror( ghost_sends );

    // specify where ghosted data go on host rank
    Kokkos::View<int*, MemorySpace> ghost_recvs( 
        Kokkos::ViewAllocateWithoutInitializing( "ghost_recvs" ),
        num_ghost_cells );
    Kokkos::View<int*, MemorySpace>::HostMirror ghost_recvs_host =
        Kokkos::create_mirror( ghost_recvs );

    // create send & recv buffer for accumulator ghosts
    _accumulator_buffer.resize(num_ghost_cells);

    int ix, iy, iz;
    int low_x = num_ghosts;
    int low_y = num_ghosts;
    int low_z = num_ghosts;
    int high_x = (nx-1)+num_ghosts;
    int high_y = (ny-1)+num_ghosts;
    int high_z = (nz-1)+num_ghosts;
    Kokkos::deep_copy( acc_exports_host, comm_rank ); // set default
    // TODO: implement neighbor array and make 3d
    // This stuff could be moved to move_p
    for ( size_t idx = 0, i = 0; i < num_cells; ++i ) {
        RANK_TO_INDEX( i, ix, iy, iz, nx+2*num_ghosts, ny+2*num_ghosts);
        if ( ix < low_x ) {
            acc_exports_host(idx) = prev_rank;
            ghost_sends_host(idx) = i;
            ghost_recvs_host(idx++) = mirror( i, nx, ny, nz, num_ghosts );
        }
        else if ( ix > high_x ) {
            acc_exports_host(idx) = next_rank;
            ghost_sends_host(idx) = i;
            ghost_recvs_host(idx++) = mirror( i, nx, ny, nz, num_ghosts );
        }
        else if ( iy < low_y || iy > high_y ||
                  iz < low_z || iz > high_z ) {
            ghost_sends_host(idx) = i;
            ghost_recvs_host(idx++) = mirror( i, nx, ny, nz, num_ghosts );
        }
    }
    Kokkos::deep_copy( acc_exports, acc_exports_host );
    Kokkos::deep_copy( ghost_sends, ghost_sends_host );
    Kokkos::deep_copy( ghost_recvs, ghost_recvs_host );
    _acc_exports = acc_exports;
    _ghost_sends = ghost_sends;
    _ghost_recvs = ghost_recvs;

    // set neighbors for accumulator topology 
    std::vector<int> accumulator_topology =
        { prev_rank, comm_rank, next_rank };
    std::sort( accumulator_topology.begin(), accumulator_topology.end() );
    auto unique_end = std::unique( 
        accumulator_topology.begin(), accumulator_topology.end() );
    accumulator_topology.resize( 
        std::distance(accumulator_topology.begin(), unique_end) );
    _accumulator_distributor = std::make_shared<Cabana::Distributor<MemorySpace>>(
        MPI_COMM_WORLD, acc_exports, accumulator_topology );
}

void accumulator_ghosts_t::scatter( accumulator_array_t accumulators )
{

    auto accumulator_values = Cabana::slice<0>( _accumulator_buffer );
    auto accumulator_cells = Cabana::slice<1>( _accumulator_buffer );

    // Ghosted cells move accumulators
    auto _fill_ghost_buffer =
        KOKKOS_LAMBDA( const size_t s, const size_t i )
    {
        int c = s*accumulator_aosoa_t::vector_length + i;
        size_t cell = _ghost_sends(c);
        accumulator_cells.access( s, i ) = _ghost_recvs(c);
        for ( size_t ii = 0; ii < ACCUMULATOR_VAR_COUNT; ++ii ) {
            for ( size_t jj = 0; jj < ACCUMULATOR_ARRAY_LENGTH; ++jj ) {
                size_t n = ii*ACCUMULATOR_ARRAY_LENGTH + jj;
                accumulator_values.access( s, i, n ) = 
                    accumulators( cell, ii, jj );
                accumulators( cell, ii, jj ) = 0;
            }
        }
    };
    Cabana::SimdPolicy<accumulator_aosoa_t::vector_length,ExecutionSpace>
        ghost_send_policy( 0, _acc_exports.size() );
    Cabana::simd_parallel_for( ghost_send_policy, 
        _fill_ghost_buffer, "fill_buffer()" );

    Cabana::migrate( *_accumulator_distributor, _accumulator_buffer );

    auto _add_ghost_accumulators = 
        KOKKOS_LAMBDA( const size_t s, const size_t i )
    {
        size_t cell = accumulator_cells.access( s, i );
        for ( size_t ii = 0; ii < ACCUMULATOR_VAR_COUNT; ++ii ) {
            for ( size_t jj = 0; jj < ACCUMULATOR_ARRAY_LENGTH; ++jj ) {
                size_t n = ii*ACCUMULATOR_ARRAY_LENGTH + jj;
                accumulators( cell, ii, jj ) +=
                    accumulator_values.access( s, i, n );
            }
        }
    };
    Cabana::SimdPolicy<accumulator_aosoa_t::vector_length,ExecutionSpace>
        ghost_recv_policy( 0, _acc_exports.size() );
    Cabana::simd_parallel_for( ghost_recv_policy, 
        _add_ghost_accumulators, "add_ghost_accumulators()" );
}

