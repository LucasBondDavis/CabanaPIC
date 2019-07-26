#ifndef pic_helper_h
#define pic_helper_h

#include "logger.h"
#include "Cabana_ExecutionPolicy.hpp" // SIMDpolicy
#include "Cabana_Parallel.hpp" // Simd parallel for

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <vector>
#include <exception>

// Converts from an index that doesn't know about ghosts to one that does
KOKKOS_INLINE_FUNCTION int allow_for_ghosts(int pre_ghost)
{

    size_t ix, iy, iz;
    RANK_TO_INDEX(pre_ghost, ix, iy, iz,
            Parameters::instance().nx,
            Parameters::instance().ny);
    //    printf("%ld\n",ix);
    int with_ghost = VOXEL(ix, iy, iz,
            Parameters::instance().nx,
            Parameters::instance().ny,
            Parameters::instance().nz,
            Parameters::instance().num_ghosts);

    return with_ghost;
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

    auto weight = particles.slice<Weight>();
    auto cell = particles.slice<Cell_Index>();

    auto _print =
        KOKKOS_LAMBDA( const int s, const int i )
        {
                printf("Struct id %d offset %d \n", s, i);
                printf("Position x %e y %e z %e \n", position_x.access(s,i), position_y.access(s,i), position_z.access(s,i) );
        };

    // TODO: How much sense does printing in parallel make???
    Cabana::SimdPolicy<particle_list_t::vector_length,ExecutionSpace>
        vec_policy( 0, particles.size() );

    //logger << "particles.numSoA() " << particles.numSoA() << std::endl;
    //logger << "particles.numSoA() " << particles.numSoA() << std::endl;

    Cabana::simd_parallel_for( vec_policy, _print, "_print()" );

    std::cout << std::endl;

}


/*
template<typename Halo_t, typename View_t>
void gather( const Halo_t& halo,
             View_t& view,
             int mpi_tag = 1002
             )
{
    // TODO: only works for 1d View
    // Check that the View is the right size.
    printf("view.size(): %lu\n", view.size());
    printf("halo.local(): %lu\n", halo.numLocal());
    printf("halo.numGhost(): %lu\n", halo.numGhost());

    if ( view.size() != halo.numLocal() + halo.numGhost() )
        throw std::runtime_error("View is the wrong size for scatter!");

    // Allocate a send buffer.
    Kokkos::View<typename View_t::data_type,
                 typename Halo_t::memory_space>
        send_buffer(
            Kokkos::ViewAllocateWithoutInitializing("halo_send_buffer"),
            halo.totalNumExport() );

    // Get the steering vector for the sends.
    auto steering = halo.getExportSteering();

    // Gather from the local data into a tuple-contiguous send buffer.
    auto gather_send_buffer_func =
        KOKKOS_LAMBDA( const std::size_t i,
                       const std::size_t j,
                       const std::size_t k )
        {
            send_buffer( i, j, k ) = view( steering(i), j, k );
        };
    // the default exec space might not be halo exec space
    int N0 = halo.totalNumExport();
    int N1 = view.extent(1);
    int N2 = view.extent(2);
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> gather_send_buffer_policy(
        {0,0,0}, {N0,N1,N2} );
    //Kokkos::RangePolicy<typename Halo_t::execution_space>
    //    gather_send_buffer_policy( 0, halo.totalNumExport() );
    Kokkos::parallel_for( "Cabana::gather::gather_send_buffer",
                          gather_send_buffer_policy,
                          gather_send_buffer_func );
    Kokkos::fence();

    // Allocate a receive buffer.
    Kokkos::View<typename View_t::data_type,
                 typename Halo_t::memory_space>
        recv_buffer(
            Kokkos::ViewAllocateWithoutInitializing("halo_recv_buffer"),
            halo.totalNumImport() );

    // Post non-blocking receives.
    int num_n = halo.numNeighbor();
    std::vector<MPI_Request> requests( num_n );
    std::pair<std::size_t,std::size_t> recv_range = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        recv_range.second =
            recv_range.first + halo.numImport(n);

        auto recv_subview = Kokkos::subview( recv_buffer, recv_range );

        MPI_Irecv( recv_subview.data(),
                   recv_subview.size() * sizeof(typename View_t::data_type),
                   MPI_BYTE,
                   halo.neighborRank(n),
                   mpi_tag,
                   halo.comm(),
                   &(requests[n]) );

        recv_range.first = recv_range.second;
    }

    // Do blocking sends.
    std::pair<std::size_t,std::size_t> send_range = { 0, 0 };
    for ( int n = 0; n < num_n; ++n )
    {
        send_range.second =
            send_range.first + halo.numExport(n);

        auto send_subview = Kokkos::subview( send_buffer, send_range );

        MPI_Send( send_subview.data(),
                  send_subview.size() * sizeof(typename View_t::data_type),
                  MPI_BYTE,
                  halo.neighborRank(n),
                  mpi_tag,
                  halo.comm() );

        send_range.first = send_range.second;
    }

    // Wait on non-blocking receives.
    std::vector<MPI_Status> status( num_n );
    const int ec =
        MPI_Waitall( requests.size(), requests.data(), status.data() );
    if ( MPI_SUCCESS != ec )
        throw std::logic_error( "Failed MPI Communication" );

    // Extract the receive buffer into the ghosted elements.
    std::size_t num_local = halo.numLocal();
    auto extract_recv_buffer_func =
        KOKKOS_LAMBDA( const std::size_t i )
        {
            std::size_t ghost_idx = i + num_local;
            // TODO: replace get tuple with something else
            view(ghost_idx) = recv_buffer(i);
        };
    Kokkos::RangePolicy<typename Halo_t::execution_space>
        extract_recv_buffer_policy( 0, halo.totalNumImport() );
    Kokkos::parallel_for( "Cabana::gather::extract_recv_buffer",
                          extract_recv_buffer_policy,
                          extract_recv_buffer_func );
    Kokkos::fence();

    // Barrier before completing to ensure synchronization.
    MPI_Barrier( halo.comm() );
}
*/

#endif // pic_helper_h
