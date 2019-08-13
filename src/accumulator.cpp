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
// TODO: doesn't work for num_ghosts > nx
int mirror( int cell, int nx, int ny, int nz, int num_ghosts ) {
    int ix, iy, iz;
    RANK_TO_INDEX( cell, ix, iy, iz, nx+2*num_ghosts, ny+2*num_ghosts );
    if ( ix < num_ghosts )
        ix += nx;
    else if ( ix > (nx-1) + num_ghosts )
        ix -= nx;
    //if ( iy < num_ghosts ) // TODO: ENABLE FOR 3d, breaks 1d
    //    iy += ny;
    //else if ( iy > (ny-1) + num_ghosts )
    //    iy -= ny;
    //if ( iz < num_ghosts )
    //    iz += nz;
    //else if ( iz > (nz-1) + num_ghosts )
    //    iz -= nz;
    return VOXEL( ix, iy, iz, nx, ny, nz, num_ghosts );
}

ghosts_t::ghosts_t(
        const size_t nx,
        const size_t ny,
        const size_t nz,
        const size_t num_ghosts,
        const size_t num_real_cells,
        const size_t num_cells,
        const int dims[3],
        MPI_Comm mpi_comm )
{ 
    // Get mpi info
    int comm_rank = -1;
    int comm_size = -1;
    int coords[3];
    _mpi_comm = mpi_comm;
    MPI_Comm_rank( _mpi_comm, &comm_rank );
    MPI_Comm_size( _mpi_comm, &comm_size );
    MPI_Cart_coords( _mpi_comm, comm_rank, 3, coords );

    // set constant for aosoa sizes
    const size_t num_ghost_cells = num_cells - num_real_cells;

    // create accumulator export ranks and ids
    Kokkos::View<int*,MemorySpace> exports(
        Kokkos::ViewAllocateWithoutInitializing( "exports" ),
        num_ghost_cells );
    Kokkos::View<int*, MemorySpace>::HostMirror exports_host =
        Kokkos::create_mirror( exports );

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
    _field_buffer.resize(num_ghost_cells);

    int ix, iy, iz;
    const int low_x = num_ghosts;
    const int low_y = num_ghosts;
    const int low_z = num_ghosts;
    const int high_x = (nx-1)+num_ghosts;
    const int high_y = (ny-1)+num_ghosts;
    const int high_z = (nz-1)+num_ghosts;
    Kokkos::deep_copy( exports_host, comm_rank ); // set default
    for ( size_t idx = 0, i = 0; i < num_cells; ++i ) {
        RANK_TO_INDEX( i, ix, iy, iz, nx+2*num_ghosts, ny+2*num_ghosts);
        ghost_sends_host(idx) = i;
        ghost_recvs_host(idx) = mirror( i, nx, ny, nz, num_ghosts );
        int cx = coords[0];
        int cy = coords[1];
        int cz = coords[2];
        if ( ix < low_x ) {
            cx = (ix + 1 - num_ghosts)/nx - 1 + coords[0];
            cx = ( cx < 0 ) ? cx + dims[0] : cx;
        }
        else if ( ix > high_x ) {
            cx = (ix - nx + num_ghosts)/nx + 1 + coords[0];
            cx = ( cx > dims[0] ) ? cx - dims[0] : cx;
        }
        //if ( iy < low_y ) {
        //    cy = (iy + 1 - num_ghosts)/ny - 1 + coords[1];
        //    cy = ( cy < 0 ) ? cy + dims[1] : cy;
        //}
        //else if ( iy > high_y ) {
        //    cy = (iy - ny + num_ghosts)/ny + 1 + coords[1];
        //    cy = ( cy > dims[1] ) ? cy - dims[1] : cy;
        //}
        //if ( iz < low_z ) {
        //    cz = (iz - 1 + num_ghosts)/nz + 1 + coords[2];
        //    cz = ( cz < 0 ) ? cz + dims[2] : cz;
        //}
        //else if ( iz > high_z ) {
        //    cz = (iz - nz + num_ghosts)/nz + 1 + coords[2];
        //    cz = ( cz > dims[2] ) ? cz - dims[2] : cz;
        //}
        if ( cx != coords[0]
          || cy != coords[1]
          || cz != coords[2] )
        {
            int dest_rank = -1;
            int c[3] = {cx, cy, cz};
            MPI_Cart_rank( _mpi_comm, c, &dest_rank );
            exports_host(idx++) = dest_rank;
        }
    }
    Kokkos::deep_copy( exports, exports_host );
    Kokkos::deep_copy( ghost_sends, ghost_sends_host );
    Kokkos::deep_copy( ghost_recvs, ghost_recvs_host );
    _exports = exports;
    _ghost_sends = ghost_sends;
    _ghost_recvs = ghost_recvs;

    // set neighbors for accumulator topology (assumes every cell is ghosted)
    std::vector<int> topology( comm_size );
    for ( int i = 0; i < comm_size; ++i )
        topology.push_back(i);
    std::sort( topology.begin(), topology.end() );
    auto unique_end = std::unique( topology.begin(), topology.end() );
    topology.resize( std::distance(topology.begin(), unique_end) );
    _distributor = std::make_shared<Cabana::Distributor<MemorySpace>>(
        _mpi_comm, _exports, topology );
}

// For periodic boundaries only
// collapses the current on the boundaries becuase they are the same
void ghosts_t::collapse_boundaries( accumulator_array_t accumulators )
{
    int dims[3];
    int wrap[3];
    int coords[3];
    MPI_Cart_get( _mpi_comm, 3, dims, wrap, coords );
    auto _collapse_boundaries =
        KOKKOS_LAMBDA( const size_t i )
        {
            size_t cell = _ghost_sends(i);
            for ( size_t j = 0; j < ACCUMULATOR_VAR_COUNT; ++j ) {
                for ( size_t k = 0; k < ACCUMULATOR_ARRAY_LENGTH; ++k ) {
                    
                }
            }
        };
}

void ghosts_t::scatter( accumulator_array_t accumulators )
{
    auto values = Cabana::slice<0>( _accumulator_buffer );
    auto cells = Cabana::slice<1>( _accumulator_buffer );

    // Ghosted cells move accumulators
    auto _fill_ghost_accumulator_buffer =
        KOKKOS_LAMBDA( const size_t s, const size_t i )
        {
            int c = s*accumulator_aosoa_t::vector_length + i;
            size_t cell = _ghost_sends(c);
            cells.access( s, i ) = _ghost_recvs(c);
            for ( size_t ii = 0; ii < ACCUMULATOR_VAR_COUNT; ++ii ) {
                for ( size_t jj = 0; jj < ACCUMULATOR_ARRAY_LENGTH; ++jj ) {
                    size_t n = ii*ACCUMULATOR_ARRAY_LENGTH + jj;
                    values.access( s, i, n ) = 
                        accumulators( cell, ii, jj );
                    accumulators( cell, ii, jj ) = 0;
                }
            }
        };
    Cabana::SimdPolicy<accumulator_aosoa_t::vector_length,ExecutionSpace>
        ghost_send_policy( 0, _exports.size() );
    Cabana::simd_parallel_for(
        ghost_send_policy, 
        _fill_ghost_accumulator_buffer,
        "fill_accumulator_buffer()"
    );

    Cabana::migrate( *_distributor, _accumulator_buffer );

    auto _add_ghost_accumulators = 
        KOKKOS_LAMBDA( const size_t s, const size_t i )
        {
            size_t cell = cells.access( s, i );
            for ( size_t ii = 0; ii < ACCUMULATOR_VAR_COUNT; ++ii ) {
                for ( size_t jj = 0; jj < ACCUMULATOR_ARRAY_LENGTH; ++jj ) {
                    size_t n = ii*ACCUMULATOR_ARRAY_LENGTH + jj;
                    accumulators( cell, ii, jj ) +=
                        values.access( s, i, n );
                }
            }
        };
    Cabana::SimdPolicy<accumulator_aosoa_t::vector_length,ExecutionSpace>
        ghost_recv_policy( 0, _exports.size() );
    Cabana::simd_parallel_for(
            ghost_recv_policy,
            _add_ghost_accumulators,
            "add_ghost_accumulators()"
    );
}

void ghosts_t::scatter( field_array_t fields )
{
    auto cells = Cabana::slice<CELLS>( _field_buffer );
    auto _ex = Cabana::slice<FIELD_EX>( _field_buffer );
    auto _jfx = Cabana::slice<FIELD_JFX>( _field_buffer );
    auto ex = Cabana::slice<FIELD_EX>( fields );
    auto jfx = Cabana::slice<FIELD_JFX>( fields );
    // Ghosted cells move accumulators
    auto _fill_ghost_buffer =
        KOKKOS_LAMBDA( const size_t s, const size_t i )
        {
            int c = s*field_array_t::vector_length + i;
            size_t cell = _ghost_sends(c);
            size_t j = cell%field_array_t::vector_length;
            size_t n = (cell-j)/field_array_t::vector_length;
            cells.access( s, i ) = _ghost_recvs(c);
            _ex.access( s, i ) = ex.access( n, j );
            _jfx.access( s, i) = jfx.access( n, j );
            ex.access( n, j ) = 0;
            jfx.access( n, j ) = 0;
        };
    Cabana::SimdPolicy<field_array_t::vector_length,ExecutionSpace>
        ghost_send_policy( 0, _exports.size() );
    Cabana::simd_parallel_for( ghost_send_policy, 
        _fill_ghost_buffer, "fill_field_buffer()" );

    Cabana::migrate( *_distributor, _field_buffer );

    auto _add_ghost_fields = 
        KOKKOS_LAMBDA( const size_t s, const size_t i )
        {
            size_t cell = cells.access( s, i ); 
            size_t j = cell%field_array_t::vector_length;
            size_t n = (cell-j)/field_array_t::vector_length;
            ex.access( n, j ) += _ex.access( s, i );
            jfx.access( n, j ) += _jfx.access( s, i );
        };
    Cabana::SimdPolicy<field_array_t::vector_length,ExecutionSpace>
        ghost_recv_policy( 0, _exports.size() );
    Cabana::simd_parallel_for( ghost_recv_policy, 
        _add_ghost_fields, "add_ghost_fields()" );
}
