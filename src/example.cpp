#include <Cabana_AoSoA.hpp>
#include <Cabana_Core.hpp>

#include <cstdlib>
#include <iostream>

#include <types.h>
#include <visualization.h>
#include <push.h>

//---------------------------------------------------------------------------//
// Helper functions.
//---------------------------------------------------------------------------//
// Function to intitialize the particles.
void initialize_particles( particle_list_t particles )
{
    auto position_x = particles.slice<PositionX>();
    auto position_y = particles.slice<PositionY>();
    auto position_z = particles.slice<PositionZ>();

    auto velocity_x = particles.slice<VelocityX>();
    auto velocity_y = particles.slice<VelocityY>();
    auto velocity_z = particles.slice<VelocityZ>();

    auto charge = particles.slice<Charge>();
    auto cell = particles.slice<Cell_Index>();

    auto _init =
        //KOKKOS_LAMBDA( const int s )
        KOKKOS_LAMBDA( const int s, const int i )
        {
            // Much more likely to vectroize and give good performance
            float counter = (float)s;
            //for ( int i = 0; i < particle_list_t::vector_length; ++i )
            //{
                // Initialize position.
                position_x.access(s,i) = 1.1 + counter;
                position_y.access(s,i) = 2.2 + counter;
                position_z.access(s,i) = 3.3 + counter;

                // Initialize velocity.
                velocity_x.access(s,i) = 0.1;
                velocity_y.access(s,i) = 0.2;
                velocity_z.access(s,i) = 0.3;

                charge.access(s,i) = 1.0;
                cell.access(s,i) = s;
                counter += 0.01;
            //}
        };
    Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
        vec_policy( 0, particles.numSoA() );
    Cabana::simd_parallel_for( vec_policy, _init, "init()" );
}

// Function to print out the data for every particle.
void print_particles( const particle_list_t particles )
{
    auto position_x = particles.slice<PositionX>();
    auto position_y = particles.slice<PositionY>();
    auto position_z = particles.slice<PositionZ>();

    auto velocity_x = particles.slice<VelocityX>();
    auto velocity_y = particles.slice<VelocityY>();
    auto velocity_z = particles.slice<VelocityZ>();

    auto charge = particles.slice<Charge>();
    auto cell = particles.slice<Cell_Index>();

    auto _print =
        KOKKOS_LAMBDA( const int s, const int i )
        {
            // Much more likely to vectroize and give good performance
            //for ( int i = 0; i < particle_list_t::vector_length; ++i )
            //{
                std::cout << "Struct id: " << s;
                std::cout << " Struct offset: " << i;
                std::cout << " Position: "
                    << position_x.access(s,i) << " "
                    << position_y.access(s,i) << " "
                    << position_z.access(s,i) << " ";
                std::cout << std::endl;

                std::cout << " Velocity "
                    << velocity_x.access(s,i) << " "
                    << velocity_y.access(s,i) << " "
                    << velocity_z.access(s,i) << " ";
                std::cout << std::endl;
            //}
        };

    Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
        vec_policy( 0, particles.numSoA() );
    Cabana::simd_parallel_for( vec_policy, _print, "_print()" );

}

void uncenter_particles(
        particle_list_t particles,
        interpolator_array_t* f0,
        real_t qdt_2mc
    )
{

    auto position_x = particles.slice<PositionX>();
    auto position_y = particles.slice<PositionY>();
    auto position_z = particles.slice<PositionZ>();

    auto velocity_x = particles.slice<VelocityX>();
    auto velocity_y = particles.slice<VelocityY>();
    auto velocity_z = particles.slice<VelocityZ>();

    auto charge = particles.slice<Charge>();
    auto cell = particles.slice<Cell_Index>();

    const real_t qdt_4mc        = -0.5*qdt_2mc; // For backward half rotate
    const real_t one            = 1.;
    const real_t one_third      = 1./3.;
    const real_t two_fifteenths = 2./15.;

    auto _uncenter =
        //KOKKOS_LAMBDA( const int s ) {
        KOKKOS_LAMBDA( const int s, const int i ) {
            //for ( int i = 0; i < particle_list_t::vector_length; ++i )
            //{
                // Grab particle properties
                real_t dx = position_x.access(s,i);   // Load position
                real_t dy = position_y.access(s,i);   // Load position
                real_t dz = position_z.access(s,i);   // Load position

                int ii = cell.access(s,i);

                // Grab interpolator
                interpolator_t& f = f0->i[ii];                          // Interpolate E

                // Calculate field values
                real_t hax = qdt_2mc*(( f.ex + dy*f.dexdy ) + dz*( f.dexdz + dy*f.d2exdydz ));
                real_t hay = qdt_2mc*(( f.ey + dz*f.deydz ) + dx*( f.deydx + dz*f.d2eydzdx ));
                real_t haz = qdt_2mc*(( f.ez + dx*f.dezdx ) + dy*( f.dezdy + dx*f.d2ezdxdy ));


                real_t cbx = f.cbx + dx*f.dcbxdx;            // Interpolate B
                real_t cby = f.cby + dy*f.dcbydy;
                real_t cbz = f.cbz + dz*f.dcbzdz;

                // Load momentum
                real_t ux = velocity_x.access(s,i);   // Load velocity
                real_t uy = velocity_y.access(s,i);   // Load velocity
                real_t uz = velocity_z.access(s,i);   // Load velocity

                real_t v0 = qdt_4mc/(float)sqrt(one + (ux*ux + (uy*uy + uz*uz)));

                // Borris push
                // Boris - scalars
                real_t v1 = cbx*cbx + (cby*cby + cbz*cbz);
                real_t v2 = (v0*v0)*v1;
                real_t v3 = v0*(one+v2*(one_third+v2*two_fifteenths));
                real_t v4 = v3/(one+v1*(v3*v3));

                v4  += v4;

                v0   = ux + v3*( uy*cbz - uz*cby );      // Boris - uprime
                v1   = uy + v3*( uz*cbx - ux*cbz );
                v2   = uz + v3*( ux*cby - uy*cbx );

                ux  += v4*( v1*cbz - v2*cby );           // Boris - rotation
                uy  += v4*( v2*cbx - v0*cbz );
                uz  += v4*( v0*cby - v1*cbx );

                ux  += hax;                              // Half advance E
                uy  += hay;
                uz  += haz;

                std::cout << " hay " << hay << " ux " << ux << std::endl;

                // Store result
                velocity_x.access(s,i) = ux;
                velocity_y.access(s,i) = uy;
                velocity_z.access(s,i) = uz;

            //}
        };

    Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
        vec_policy( 0, particles.numSoA() );
    Cabana::simd_parallel_for( vec_policy, _uncenter, "uncenter()" );
}

void initialize_interpolator(interpolator_array_t* f)
{
    for (size_t i = 0; i < f->size; i++)
    {
        // Current one
        auto& f_ = f->i[i];

        // Throw in some place holder values
#ifdef ELECTRO_
        f_.ex = 0.01;
#endif
        f_.dexdy = 0.02;
        f_.dexdz = 0.03;
        f_.d2exdydz = 0.04;
        f_.ey = 0.05;
        f_.deydz = 0.06;
        f_.deydx = 0.07;
        f_.d2eydzdx = 0.08;
        f_.ez = 0.09;
        f_.dezdx = 0.10;
        f_.dezdy = 0.11;
        f_.d2ezdxdy = 0.12;
        f_.cbx = 0.13;
        f_.dcbxdx = 0.14;
        f_.cby = 0.15;
        f_.dcbydy = 0.16;
        f_.cbz = 0.17;
        f_.dcbzdz = 0.18;
    }
}

void write_vis(particle_list_t particles, Visualizer& vis, size_t step)
{

  size_t total_num_particles = particles.size();

  /*
  for (unsigned int sn = 0; sn < species.size(); sn++)
  {
    int particle_count = species[sn].num_particles;
    total_num_particles += particle_count;
  }
  */

  vis.write_header(total_num_particles, step);

  //for (unsigned int sn = 0; sn < species.size(); sn++)
  //{
    //auto particles_accesor = get_particle_accessor(m, species[sn].key);
    vis.write_particles_position(particles);
  //}

  vis.write_cell_types(total_num_particles);

  vis.pre_scalars(total_num_particles);
  vis.write_particles_property_header("weight", total_num_particles);

  //for (unsigned int sn = 0; sn < species.size(); sn++)
  //{
    //auto particles_accesor = get_particle_accessor(m, species[sn].key);
    vis.write_particles_w(particles);
  //}
  //*/

  vis.write_particles_property_header("species", total_num_particles);

  //for (unsigned int sn = 0; sn < species.size(); sn++)
  //{
    //auto particles_accesor = get_particle_accessor(m, species[sn].key);
    vis.write_particles_sp(particles, 1);
  //}
  vis.finalize();

}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    // Initialize the kokkos runtime.
    Cabana::initialize( argc, argv );

    // Declare a number of particles.
    int num_particle = 45;

    // Create the particle list.
    particle_list_t particles( num_particle );

    // Initialize particles.
    initialize_particles( particles );

    grid_t* g = new grid();

    real_t qdt_2mc = 1.0f;
    real_t cdt_dx = 1.0f;
    real_t cdt_dy = 1.0f;
    real_t cdt_dz = 1.0f;
    real_t qsp = 1.0f;

    std::cout << "Initial:" << std::endl;
    print_particles( particles );

    const size_t num_steps = 10;
    const size_t num_cells = 1;

    // OLD WAY TO CREATE DATA
    // If we force ii = 0 for all particles, these can be 1 big?
    //interpolator_array_t* f = new interpolator_array_t(num_cells);
    //accumulator_array_t* a = new accumulator_array_t(num_cells);

    // NEW CABANA WAY
    interpolator_array_t f(num_cells);
    accumulator_array_t a(num_cells);

    initialize_interpolator(f);

    Visualizer vis;

    for (size_t step = 0; step < num_steps; step++)
    {
        std::cout << "Step " << step << std::endl;

        // Move
        //uncenter_particles( particles, f, qdt_2mc);
        push(
            particles,
            f,
            qdt_2mc,
            cdt_dx,
            cdt_dy,
            cdt_dz,
            qsp,
            a,
            g
        );

        // Print particles.
        print_particles( particles );

        write_vis(particles, vis, step);
    }

    // TODO: delete kokkos views/cabana data
    // Finalize.
    Cabana::finalize();
    return 0;
}

//---------------------------------------------------------------------------//
