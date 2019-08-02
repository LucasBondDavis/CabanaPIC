#include <Cabana_AoSoA.hpp>
#include <Cabana_Core.hpp>
#include <Cabana_Sort.hpp> // is this needed if we already have core?

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include <mpi.h>

#include "types.h"
#include "helpers.h"
#include "simulation_parameters.h"

#include "initializer.h"

#include "fields.h"
#include "accumulator.h"
#include "interpolator.h"

#include "push.h"

#include "visualization.h"


const int num_com_round = 3;
void boundary_p(particle_list_t& particles_in)
{
    for (int i = 0; i < num_com_round; i++)
    {
        // Send

        // Recv

        // Call move_p on new particles (implies current accumulation)
    }
}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    // Initialize the MPI & kokkos runtimes
    MPI_Init( &argc, &argv );
    Cabana::initialize( argc, argv );

    printf ("#On Kokkos execution space %s\n",
            typeid (Kokkos::DefaultExecutionSpace).name ());

    int comm_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    int comm_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );
    // Cabana scoping block
    {

        Visualizer vis;

        // Initialize input deck params.

        // num_cells (without ghosts), num_particles_per_cell
        size_t npc = 4000;
        Initializer::initialize_params(32, npc);

        // Cache some values locally for printing
        const size_t nx = Parameters::instance().nx;
        const size_t ny = Parameters::instance().ny;
        const size_t nz = Parameters::instance().nz;
        const size_t num_ghosts = Parameters::instance().num_ghosts;
        const size_t num_real_cells = Parameters::instance().num_real_cells;
        const size_t num_cells = Parameters::instance().num_cells;
        const size_t num_ghost_cells = num_cells - num_real_cells;
        const size_t num_particles = Parameters::instance().num_particles;
        const int prev_rank = ( comm_rank == 0 ) ? comm_size - 1 : comm_rank - 1;
        const int next_rank = ( comm_rank == comm_size - 1 ) ? 0 : comm_rank + 1;
        real_t dxp = 2.f/(npc);

        // Define some consts
        const real_t dx = Parameters::instance().dx;
        const real_t dy = Parameters::instance().dy;
        const real_t dz = Parameters::instance().dz;
        real_t dt   = Parameters::instance().dt;
        real_t c    = Parameters::instance().c;
        real_t me   = Parameters::instance().me;
        real_t n0   = Parameters::instance().n0;
        real_t ec   = Parameters::instance().ec;
        real_t Lx   = Parameters::instance().len_x;
        real_t Ly   = Parameters::instance().len_y;
        real_t Lz   = Parameters::instance().len_z;
        real_t nppc = Parameters::instance().NPPC;
        real_t Npe  = n0*Lx*Ly*Lz;
        real_t Ne= nppc*nx*ny*nz;
        real_t qsp = ec;
        real_t qdt_2mc = qsp*dt/(2*me*c);
        real_t cdt_dx = c*dt/dx;
        real_t cdt_dy = c*dt/dy;
        real_t cdt_dz = c*dt/dz;

        real_t frac = 1.0f;
        real_t we = Npe/Ne;

        // Create the particle list.
        particle_list_t particles( num_particles );
        //logger << "size " << particles.size() << std::endl;
        //logger << "numSoA " << particles.numSoA() << std::endl;

        // Initialize particles.
        Initializer::initialize_particles( particles, nx, ny, nz, dxp, npc, we );

        grid_t* grid = new grid_t();

        // Print initial particle positions
        //logger << "Initial:" << std::endl;
        //print_particles( particles );

        // Allocate Cabana Data
        interpolator_array_t interpolators(num_cells);

        accumulator_array_t accumulators("Accumulator View", num_cells);
        auto scatter_add = Kokkos::Experimental::create_scatter_view(accumulators);

        field_array_t fields(num_cells);

        Initializer::initialize_interpolator(interpolators);

        // Can obviously supply solver type at compile time
        //Field_Solver<EM_Field_Solver> field_solver;
        Field_Solver<ES_Field_Solver_1D> field_solver;

        // Grab some global values for use later
        const Boundary boundary = Parameters::instance().BOUNDARY_TYPE;

        //logger << "nx " << Parameters::instance().nx << std::endl;
        //logger << "num_particles " << num_particles << std::endl;
        //logger << "num_cells " << num_cells << std::endl;
        //logger << "Actual NPPC " << Parameters::instance().NPPC << std::endl;

        const real_t px =  (nx>1) ? frac*c*dt/dx : 0;
        const real_t py =  (ny>1) ? frac*c*dt/dy : 0;
        const real_t pz =  (nz>1) ? frac*c*dt/dz : 0;

        // create slice for particles distributor exports
        auto particle_exports = particles.slice<Comm_Rank>();

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
        accumulator_aosoa_t accumulator_buffer(
            "accumulator_buffer", num_ghost_cells*2 );
        auto accumulator_slice = Cabana::slice<0>( accumulator_buffer );


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
                ghost_sends_host(idx) = i + num_ghost_cells;
                ghost_recvs_host(idx++) = mirror( i, nx, ny, nz, num_ghosts );
            }
            else if ( ix > high_x ) {
                acc_exports_host(idx) = next_rank;
                ghost_sends_host(idx) = i + num_ghost_cells;
                ghost_recvs_host(idx++) = mirror( i, nx, ny, nz, num_ghosts );
            }
            else if ( iy < low_y || iy > high_y ||
                      iz < low_z || iz > high_z ) {
                ghost_sends_host(idx) = i + num_ghost_cells;
                ghost_recvs_host(idx++) = mirror( i, nx, ny, nz, num_ghosts );
            }
        }
        Kokkos::deep_copy( acc_exports, acc_exports_host );
        Kokkos::deep_copy( ghost_sends, ghost_sends_host );
        Kokkos::deep_copy( ghost_recvs, ghost_recvs_host );

        // set neighbors for accumulator topology 
        std::vector<int> accumulator_topology =
            { prev_rank, comm_rank, next_rank };
        std::sort( accumulator_topology.begin(), accumulator_topology.end() );
        auto unique_end = std::unique( 
            accumulator_topology.begin(), accumulator_topology.end() );
        accumulator_topology.resize( 
            std::distance(accumulator_topology.begin(), unique_end) );
        auto accumulator_halo = Cabana::Halo<MemorySpace>(
            MPI_COMM_WORLD, num_ghost_cells, ghost_sends, acc_exports, accumulator_topology );

        // simulation loop

        const size_t num_steps = Parameters::instance().num_steps;
        //printf( "#***********************************************\n" );
        //printf( "#num_step = %ld\n" , num_steps );
        //printf( "#Lx/de = %f\n" , Lx );
        //printf( "#Ly/de = %f\n" , Ly );
        //printf( "#Lz/de = %f\n" , Lz );
        //printf( "#nx = %ld\n" , nx );
        //printf( "#ny = %ld\n" , ny );
        //printf( "#nz = %ld\n" , nz );
        //printf( "#nppc = %lf\n" , nppc );
        //printf( "#Ne = %lf\n" , Ne );
        //printf( "#dt*wpe = %f\n" , dt );
        //printf( "#dx/de = %f\n" , Lx/(nx) );
        //printf( "#dy/de = %f\n" , Ly/(ny) );
        //printf( "#dz/de = %f\n" , Lz/(nz) );
        //printf( "#n0 = %f\n" , n0 );
        //printf( "#***********************************************\n" );

        for (size_t step = 0; step < num_steps; step++)
        {
            //     //std::cout << "Step " << step << std::endl;
            // Convert fields to interpolators

            load_interpolator_array(fields, interpolators, nx, ny, nz, num_ghosts);

            clear_accumulator_array(fields, accumulators, nx, ny, nz);
            //     auto keys = particles.slice<Cell_Index>();
            //     auto bin_data = Cabana::sortByKey( keys );

            // Move
            push(
                    particles,
                    interpolators,
                    qdt_2mc,
                    cdt_dx,
                    cdt_dy,
                    cdt_dz,
                    qsp,
                    scatter_add,
                    grid,
                    nx,
                    ny,
                    nz,
                    num_ghosts,
                    boundary
                );

            // migrate particles across mpi ranks
            auto particle_exports = particles.slice<Comm_Rank>();
            auto particle_distributor = Cabana::Distributor<MemorySpace>(
                MPI_COMM_WORLD, particle_exports );
            Cabana::migrate( particle_distributor, particles );

            // NOTE: (deprecated) move particles to ghost cells instead of this
            //auto cell = particles.slice<Cell_Index>();
            //auto x_vel = particles.slice<VelocityX>();
            //auto disp_x = particles.slice<DispX>();
            //auto _move_p =
            //    KOKKOS_LAMBDA( const int s, const int i ) {
            //        //while ( disp_x.access(s,i) > 0 ) { // should termiante after 4 iterations
            //        // TODO: This is dangerous...
            //        if ( disp_x.access(s,i) != 0.0 ) {
            //            int ii = cell.access(s,i);
            //            auto weights = particles.slice<Weight>();
            //            real_t q = weights.access(s,i)*qsp;
            //            move_p( scatter_add, particles, q, grid, s, i, nx, ny, nz,
            //                    num_ghosts, boundary );
            //        }
            //    };
            //Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
            //    vec_policy( 0, particles.size() );
            //Cabana::simd_parallel_for( vec_policy, _move_p, "move_p" );

            Kokkos::Experimental::contribute(accumulators, scatter_add);

            // Only reset the data if these two are not the same arrays
            scatter_add.reset_except(accumulators);

            //boundary_p();
///////////////////////////////////////////////////////////////////////////////////////////////////
            std::cout << std::endl;
            int tx, ty, tz;
            for ( int zz = 0; zz < num_cells; zz++) {
              RANK_TO_INDEX(zz, tx, ty, tz, nx+2, ny+2);
              if ( ty == 1 && tz == 1 ) {
                std::cout << zz << " " << tx << " " << accumulators(zz,0,0) << std::endl;
              }
            }
            std::cout << std::endl;
///////////////////////////////////////////////////////////////////////////////////////////////////

            // Ghosted cells move accumulators
            auto _fill_ghost_buffer =
                KOKKOS_LAMBDA( const size_t s, const size_t i )
            {
                int c = s*accumulator_aosoa_t::vector_length + i;
                size_t cell = ghost_sends(c) - ghost_sends.size();
                for ( size_t ii = 0; ii < ACCUMULATOR_VAR_COUNT; ++ii ) {
                    for ( size_t jj = 0; jj < ACCUMULATOR_ARRAY_LENGTH; ++jj ) {
                        size_t n = ii*ACCUMULATOR_ARRAY_LENGTH + jj;
                        accumulator_slice.access( s, i, n ) = 
                            accumulators( cell, ii, jj );
                        accumulators( cell, ii, jj ) = 0;
                    }
                }
            };
            Cabana::SimdPolicy<accumulator_aosoa_t::vector_length,ExecutionSpace>
                ghost_send_policy( 0, num_ghost_cells );
            Cabana::simd_parallel_for( ghost_send_policy, 
                _fill_ghost_buffer, "fill_buffer()" );

            Cabana::gather( accumulator_halo, accumulator_buffer );

            auto _add_ghost_accumulators = 
                KOKKOS_LAMBDA( const size_t s, const size_t i )
            {
                int c = s*accumulator_aosoa_t::vector_length + i - ghost_recvs.size();
                size_t cell = ghost_recvs(c);
                for ( size_t ii = 0; ii < ACCUMULATOR_VAR_COUNT; ++ii ) {
                    for ( size_t jj = 0; jj < ACCUMULATOR_ARRAY_LENGTH; ++jj ) {
                        size_t n = ii*ACCUMULATOR_ARRAY_LENGTH + jj;
                        accumulators( cell, ii, jj ) +=
                            accumulator_slice.access( s, i, n );
                    }
                }
///////////////////////////////////////////////////////////////////////////////////////////////////
//if ( cell > 70 && cell < 90 ) {
std::cout << "cell: " << cell << ", s: " << s << ", i: " << i
    << "\nacc: " << accumulators(cell,0,0)
    << "\nslice: " << accumulator_slice.access(s,i,0)
    << std::endl;
//}
///////////////////////////////////////////////////////////////////////////////////////////////////
            };
            Cabana::SimdPolicy<accumulator_aosoa_t::vector_length,ExecutionSpace>
                ghost_recv_policy( num_ghost_cells, 2*num_ghost_cells );
            Cabana::simd_parallel_for( ghost_recv_policy, 
                _add_ghost_accumulators, "add_ghost_accumulators()" );

            // Map accumulator current back onto the fields
            unload_accumulator_array(fields, accumulators, nx, ny, nz, num_ghosts, dx, dy, dz, dt);

            //     // Half advance the magnetic field from B_0 to B_{1/2}
            //     field_solver.advance_b(fields, px, py, pz, nx, ny, nz);

            // Advance the electric field from E_0 to E_1
            field_solver.advance_e(fields, px, py, pz, nx, ny, nz);
            MPI_Barrier( MPI_COMM_WORLD );

            //     // Half advance the magnetic field from B_{1/2} to B_1
            //     field_solver.advance_b(fields, px, py, pz, nx, ny, nz);

            //     // Print particles.
            //     print_particles( particles );

            //     // Output vis
            //     vis.write_vis(particles, step);
            // reduce over mpi ranks
            float e_energy = field_solver.e_energy(fields, px, py, pz, nx, ny, nz);
            float total_e_energy = -1;
            MPI_Reduce( &e_energy, &total_e_energy, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD );
            if ( comm_rank == 0 )
                printf( "%ld %f, %f\n", step, step*dt, total_e_energy );
            //if ( comm_rank == 0 )
            //    printf("time:%ld %f, %f\n",step, step*dt, field_solver.e_energy(fields, px, py, pz, nx, ny, nz));
        }

    } // End Scoping block

    printf("#Good!\n");
    // Finalize.
    Cabana::finalize();
    MPI_Finalize();
    return 0;
}

//---------------------------------------------------------------------------//
//

////// Known Possible Improvements /////
// I pass nx/ny/nz round a lot more than I could

