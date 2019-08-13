#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>

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

    // Create Cartesian grid topology
    int comm_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    MPI_Comm grid_comm;
    const int dims[3] = {comm_size, 1, 1}; // (nx, ny, nz)
    const int wrap[3] = {1, 1, 1}; // periodic in all dimensions

    MPI_Cart_create( MPI_COMM_WORLD, 3, dims, wrap, false, &grid_comm );

    printf ("#On Kokkos execution space %s\n",
            typeid (Kokkos::DefaultExecutionSpace).name ());

    {

        Visualizer vis;

        // Initialize input deck params.

        // num_cells (without ghosts), num_particles_per_cell
        size_t npc = 4000;
        Initializer::initialize_params(64, npc);

        // Cache some values locally for printing
        const size_t nx = Parameters::instance().nx;
        const size_t ny = Parameters::instance().ny;
        const size_t nz = Parameters::instance().nz;
        const size_t num_ghosts = Parameters::instance().num_ghosts;
        const size_t num_real_cells = Parameters::instance().num_real_cells;
        const size_t num_cells = Parameters::instance().num_cells;
        const size_t num_particles = Parameters::instance().num_particles;
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

        ghosts_t cell_ghosts(
            nx, ny, nz, num_ghosts, num_real_cells, num_cells, dims, grid_comm );

        field_array_t fields(num_cells);

        Initializer::initialize_interpolator(interpolators);

        // Can obviously supply solver type at compile time
        //Field_Solver<EM_Field_Solver> field_solver;
        Field_Solver<ES_Field_Solver> field_solver;

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
                    boundary,
                    grid_comm
                );

            // migrate particles across mpi ranks
            auto particle_exports = particles.slice<Comm_Rank>();
            auto particle_distributor = Cabana::Distributor<MemorySpace>(
                grid_comm, particle_exports );
            Cabana::migrate( particle_distributor, particles );

            Kokkos::Experimental::contribute(accumulators, scatter_add);

            // Only reset the data if these two are not the same arrays
            scatter_add.reset_except(accumulators);

            // Ghosted cells move accumulators
            cell_ghosts.scatter(accumulators);

            // Map accumulator current back onto the fields
            unload_accumulator_array(fields, accumulators, nx, ny, nz, num_ghosts, dx, dy, dz, dt);

            // TODO: I don't know where exactly the field scatter should go
            cell_ghosts.scatter(fields);

            //     // Half advance the magnetic field from B_0 to B_{1/2}
            //     field_solver.advance_b(fields, px, py, pz, nx, ny, nz);

            // Advance the electric field from E_0 to E_1
            field_solver.advance_e(fields, px, py, pz, nx, ny, nz);

            //     // Half advance the magnetic field from B_{1/2} to B_1
            //     field_solver.advance_b(fields, px, py, pz, nx, ny, nz);

            //     // Print particles.
            //     print_particles( particles );

            //     // Output vis
            //     vis.write_vis(particles, step);
            // reduce over mpi ranks
            float e_energy = field_solver.e_energy(fields, px, py, pz, nx, ny, nz);
            float total_e_energy = -1;
            MPI_Reduce( &e_energy, &total_e_energy, 1, MPI_FLOAT, MPI_SUM, 0, grid_comm );
            printf( "%ld %f, %f\n", step, step*dt, total_e_energy );
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

            // NOTE: (deprecated) this could be moved to boundary_p()
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

            //std::cout << std::endl;
            //int tx, ty, tz;
            //for ( int zz = 0; zz < num_cells; zz++) {
            //  RANK_TO_INDEX(zz, tx, ty, tz, nx+2, ny+2);
            //  if ( ty == 1 && tz == 1 ) {
            //    std::cout << zz << " " << tx << " " << accumulators(zz,0,0) << std::endl;
            //  }
            //}
            //std::cout << std::endl;

////// Known Possible Improvements /////
// I pass nx/ny/nz round a lot more than I could

