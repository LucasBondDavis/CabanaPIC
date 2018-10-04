#include <Cabana_AoSoA.hpp>
#include <Cabana_Core.hpp>
//#include <Cabana_MemberDataTypes.hpp>
//#include <Cabana_Serial.hpp>

#include <cstdlib>
#include <iostream>

#include <interpolator.h>

#define real_t float

//---------------------------------------------------------------------------//
// Define particle data.
//---------------------------------------------------------------------------//
// Inner array size (the size of the arrays in the structs-of-arrays).
#ifndef VLEN
#define VLEN 16
#endif
const std::size_t array_size = VLEN;

using MemorySpace = Cabana::HostSpace;
using ExecutionSpace = Kokkos::Serial;
using parallel_algorithm_tag = Cabana::StructParallelTag;

// User field enumeration. These will be used to index into the data set. Must
// start at 0 and increment contiguously.
//
// NOTE: Users don't have to make this enum (or some other set of integral
// constants) but it is a nice way to provide meaning to the different data
// types and values assigned to the particles.
//
// NOTE: These enums are also ordered in the same way as the data in the
// template parameters below.
enum UserParticleFields
{
    PositionX = 0,
    PositionY,
    PositionZ,
    VelocityX,
    VelocityY,
    VelocityZ,
    Charge,
    Cell_Index,
};

// Designate the types that the particles will hold.
using ParticleDataTypes =
Cabana::MemberTypes<
    float,                        // (0) x-position
    float,                        // (1) y-position
    float,                        // (2) z-position
    float,                        // (3) x-velocity
    float,                        // (4) y-velocity
    float,                        // (5) z-velocity
    float,                        // (6) charge
    int                           // (7) Cell index
>;

// Set the type for the particle AoSoA.
using particle_list_t =
    Cabana::AoSoA<ParticleDataTypes,MemorySpace,array_size>;

//---------------------------------------------------------------------------//
// Helper functions.
//---------------------------------------------------------------------------//
// Function to intitialize the particles.
void initializeParticles( particle_list_t particles )
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
        KOKKOS_LAMBDA( const int s )
        {
            // Much more likely to vectroize and give good performance
            float counter = (float)s;
            for ( int i = 0; i < particle_list_t::vector_length; ++i )
            {
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
            }
        };
    Cabana::RangePolicy<particle_list_t::vector_length,ExecutionSpace>
        vec_policy( 0, particles.numSoA() );
    Cabana::parallel_for( vec_policy, _init, parallel_algorithm_tag() );
}

// Function to print out the data for every particle.
void printParticles( const particle_list_t particles )
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
        KOKKOS_LAMBDA( const int s )
        {
            // Much more likely to vectroize and give good performance
            for ( int i = 0; i < particle_list_t::vector_length; ++i )
            {
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
            }
        };

    Cabana::RangePolicy<particle_list_t::vector_length,ExecutionSpace>
        vec_policy( 0, particles.numSoA() );
    Cabana::parallel_for( vec_policy, _print, parallel_algorithm_tag() );

}

void uncenter_particles(
        particle_list_t particles,
        interpolator_array_t* f0,
        real_t qdt_2mc )
{
    /*
    const real_t qdt_4mc        = -0.5*qdt_2mc; // For backward half rotate
    const real_t one            = 1.;
    const real_t one_third      = 1./3.;
    const real_t two_fifteenths = 2./15.;

    //int first, ii, n;

    for ( auto idx = particles.begin();
            idx != particles.end();
            ++idx )
    {
        // Grab particle properties
        real_t dx = particles.get<PositionX>(idx);                            // Load position
        real_t dy = particles.get<PositionY>(idx);
        real_t dz = particles.get<PositionZ>(idx);
        int ii = particles.get<Cell_Index>(idx);

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
        real_t ux = particles.get<VelocityX>(idx);
        real_t uy = particles.get<VelocityY>(idx);
        real_t uz = particles.get<VelocityZ>(idx);

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
        particles.get<VelocityX>(idx) = ux;
        particles.get<VelocityY>(idx) = uy;
        particles.get<VelocityZ>(idx) = uz;

    }
*/
}

void initialize_interpolator(interpolator_array_t* f)
{
    for (size_t i = 0; i < f->size; i++)
    {
        // Current one
        auto& f_ = f->i[i];

        // Throw in some place holder values
        f_.ex = 0.01;
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
    initializeParticles( particles );

    // Uncenter Particles
    real_t qdt_2md = 1.0f;

    std::cout << "Initial:" << std::endl;
    printParticles( particles );

    size_t num_steps = 10;

    // If we force ii = 0 for all particles, this can be 1 big?
    interpolator_array_t* f = new interpolator_array_t(1);

    initialize_interpolator(f);

    for (size_t step = 0; step < num_steps; step++)
    {
        std::cout << "Step " << step << std::endl;
        // Move
        uncenter_particles( particles, f, qdt_2md);

        // Print particles.
        printParticles( particles );
    }

    // Finalize.
    Cabana::finalize();
    return 0;
}

//---------------------------------------------------------------------------//
