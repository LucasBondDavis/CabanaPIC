#ifndef pic_move_p_h
#define pic_move_p_h

#include <types.h>


// I make no claims that this is a sensible way to do this.. I just want it working ASAP
// THIS DEALS WITH GHOSTS ITSELF
KOKKOS_INLINE_FUNCTION int detect_leaving_domain( size_t face, size_t nx, size_t ny, size_t nz, size_t ix, size_t iy, size_t iz, size_t num_ghosts )
{

    //RANK_TO_INDEX(ii, ix, iy, iz, (nx+(2*num_ghosts)), (ny+(2*num_ghosts)));
    //std::cout << "i " << ii << " ix " << ix << " iy " << iy << " iz " << iz << std::endl;

    //printf("nx,ny,nz=%ld,%ld,%ld, i=%ld, ix=%ld, iy=%ld, iz=%ld\n",nx,ny,nz,ii,ix,iy,iz);

    int leaving = -1;

    if (ix == 0)
    {
        leaving = 0;
    }

    if (iy == 0)
    {
        leaving = 1;
    }

    if (iz == 0)
    {
        leaving = 2;
    }

    if (ix == nx+1)
    {
        leaving = 3;
    }

    if (iy == ny+1)
    {
        leaving = 4;
    }

    if (iz == nz+1)
    {
        leaving = 5;
    }

    return leaving;
}


// TODO: add namespace etc?
// TODO: port this to cabana syntax
template<typename T>
KOKKOS_INLINE_FUNCTION int move_p(
        T& a0, // TODO: does this need to be const
        particle_list_t particles,
        particle_mover_t pm,
        real_t q,
        const grid_t* g,
        const size_t s,
        const size_t i,
        const size_t nx,
        const size_t ny,
        const size_t nz,
        const size_t num_ghosts,
        const Boundary boundary
    )
{
    // Get local MPI rank and world_size
    int comm_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    int comm_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &comm_rank );

    auto position_x = particles.slice<PositionX>();
    auto position_y = particles.slice<PositionY>();
    auto position_z = particles.slice<PositionZ>();

    auto velocity_x = particles.slice<VelocityX>();
    auto velocity_y = particles.slice<VelocityY>();
    auto velocity_z = particles.slice<VelocityZ>();

    auto weight = particles.slice<Weight>();
    auto cell = particles.slice<Cell_Index>();
    auto mpi_rank = particles.slice<Comm_Rank>();

    auto _asa = a0.access(); // accumulator scatter access

    /* // Kernel variables */
    real_t s_dir[3];
    real_t v0, v1, v2, v3; //, v4, v5;
    size_t axis, face;
    /* //particle_t* p = p0 + pm->i; */
    /* //int index = pm->i; */

    //q = qsp * weight.access(s, i);

    for(;;)
    {
        /*
           s_midx = p->dx;
           s_midy = p->dy;
           s_midz = p->dz;
           */

        float s_midx = position_x.access(s, i);
        float s_midy = position_y.access(s, i);
        float s_midz = position_z.access(s, i);

        float s_dispx = pm.dispx;
        float s_dispy = pm.dispy;
        float s_dispz = pm.dispz;

        s_dir[0] = (s_dispx>0) ? 1 : -1;
        s_dir[1] = (s_dispy>0) ? 1 : -1;
        s_dir[2] = (s_dispz>0) ? 1 : -1;

        // Compute the twice the fractional distance to each potential
        // streak/cell face intersection.
        v0 = (s_dispx==0) ? 3.4e38 : (s_dir[0]-s_midx)/s_dispx;
        v1 = (s_dispy==0) ? 3.4e38 : (s_dir[1]-s_midy)/s_dispy;
        v2 = (s_dispz==0) ? 3.4e38 : (s_dir[2]-s_midz)/s_dispz;

        // Determine the fractional length and axis of current streak. The
        // streak ends on either the first face intersected by the
        // particle track or at the end of the particle track.
        //
        //   axis 0,1 or 2 ... streak ends on a x,y or z-face respectively
        //   axis 3        ... streak ends at end of the particle track
        /**/      v3=2,  axis=3;
        if(v0<v3) v3=v0, axis=0;
        if(v1<v3) v3=v1, axis=1;
        if(v2<v3) v3=v2, axis=2;
        v3 *= 0.5;

        // Compute the midpoint and the normalized displacement of the streak
        s_dispx *= v3;
        s_dispy *= v3;
        s_dispz *= v3;
        s_midx += s_dispx;
        s_midy += s_dispy;
        s_midz += s_dispz;

        // Accumulate the streak.  Note: accumulator values are 4 times
        // the total physical charge that passed through the appropriate
        // current quadrant in a time-step
        //v5 = q*s_dispx*s_dispy*s_dispz*(1./3.);

        int ii = cell.access(s, i);

        //a = (float *)(a0 + ii);
        _asa(ii,accumulator_var::jx, 0) += q*s_dispx;
        _asa(ii,accumulator_var::jx, 1) += 0.0;
        _asa(ii,accumulator_var::jx, 2) += 0.0;
        _asa(ii,accumulator_var::jx, 3) += 0.0;

        // #   define accumulate_j(X,Y,Z, offset)                                    \
        //         v4  = q*s_disp##X;    /* v2 = q ux                            */  \
        //         v1  = v4*s_mid##Y;    /* v1 = q ux dy                         */  \
        //         v0  = v4-v1;          /* v0 = q ux (1-dy)                     */  \
        //         v1 += v4;             /* v1 = q ux (1+dy)                     */  \
        //         v4  = 1+s_mid##Z;     /* v4 = 1+dz                            */  \
        //         v2  = v0*v4;          /* v2 = q ux (1-dy)(1+dz)               */  \
        //         v3  = v1*v4;          /* v3 = q ux (1+dy)(1+dz)               */  \
        //         v4  = 1-s_mid##Z;     /* v4 = 1-dz                            */  \
        //         v0 *= v4;             /* v0 = q ux (1-dy)(1-dz)               */  \
        //         v1 *= v4;             /* v1 = q ux (1+dy)(1-dz)               */  \
        //         v0 += v5;             /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */  \
        //         v1 -= v5;             /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */  \
        //         v2 -= v5;             /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */  \
        //         v3 += v5;             /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */  \
        // 	a0(ii,offset+0) += v0;						\
        // 	a0(ii,offset+1) += v1;						\
        // 	a0(ii,offset+2) += v2;						\
        // 	a0(ii,offset+3) += v3;
        //         accumulate_j(x,y,z, 0); //a += 4;
        //         accumulate_j(y,z,x, 4); //a += 4;
        //         accumulate_j(z,x,y, 8);
        // #   undef accumulate_j

        // Compute the remaining particle displacment
        //if (ii == (nx+num_ghosts*2)*4+nx+num_ghosts-1 )
        //{
        //    if ( comm_rank == comm_size-1 ) {
        //        std::cout << " move_p: s " << s << " i " << i <<
        //            " disp_x(s,i) " <<
        //            disp_x.access(s,i) <<
        //            std::endl;
        //    }
        //}
        pm.dispx -= s_dispx;
        pm.dispy -= s_dispy;
        pm.dispz -= s_dispz;

        //printf("%d %d, %d, %f %f",s, i, ii, position_x.access(s, i),position_x.access(s, i));
        // Compute the new particle offset
        position_x.access(s, i) += s_dispx+s_dispx;
        position_y.access(s, i) += s_dispy+s_dispy;
        position_z.access(s, i) += s_dispz+s_dispz;

        //printf(" %f\n",position_x.access(s, i));

        // If an end streak, return success (should be ~50% of the time)

        if( axis==3 ) break;

        // Determine if the particle crossed into a local cell or if it
        // hit a boundary and convert the coordinate system accordingly.
        // Note: Crossing into a local cell should happen ~50% of the
        // time; hitting a boundary is usually a rare event.  Note: the
        // entry / exit coordinate for the particle is guaranteed to be
        // +/-1 _exactly_ for the particle.

        v0 = s_dir[axis];

        // TODO: do branching based on axis

        //(&(p->dx))[axis] = v0; // Avoid roundoff fiascos--put the particle

        // TODO: this conditional could be better
        if (axis == 0) position_x.access(s, i) = v0;
        if (axis == 1) position_y.access(s, i) = v0;
        if (axis == 2) position_z.access(s, i) = v0;

        // _exactly_ on the boundary.
        face = axis;
        if( v0>0 ) face += 3;

        size_t ix, iy, iz;
        //RANK_TO_INDEX(ii, ix, iy, iz, (nx-1+(2*num_ghosts)), (ny-1+(2*num_ghosts)));
        ix = ii-((nx+2)*(ny+2)+(nx+2)); //ii-12;
        iy = 1;
        iz = 1;

        if (face == 0) { ix--; }
        if (face == 1) { iy--; }
        if (face == 2) { iz--; }
        if (face == 3) { ix++; }
        if (face == 4) { iy++; }
        if (face == 5) { iz++; }

        int is_leaving_domain = detect_leaving_domain(face, nx, ny, nz, ix, iy, iz, num_ghosts);
        if (is_leaving_domain >= 0) {
            //std::cout << s << ", " << i << " leaving on " << face << std::endl;

                 //std::cout <<
                 //" x " << position_x.access(s,i) <<
                 //" y " << position_y.access(s,i) <<
                 //" z " << position_z.access(s,i) <<
                 //" cell " << cell.access(s,i) <<
                 //std::endl;
            // TODO: make neighbor list for 3d (look at VPIC)

            if ( boundary == Boundary::Periodic)
            {
                //std::cout << "face" << std::endl;
                // If we hit the periodic boundary, try and put the article in the right place

                // TODO: we can do this in 1d just fine

                //siz_t ix, iy, iz;

                //RANK_TO_INDEX(ii, ix, iy, iz, (nx-1+(2*num_ghosts)), (ny-1+(2*num_ghosts)));
                /* ix = ii-12; */
                /* iy = 1; */
                /* iz = 1; */
                // TODO boundaries change if we allow particles into ghost cells

                if (is_leaving_domain == 0) { // -1 on x face
                    //ix = (nx-1) + num_ghosts; // happens after pass now
                    mpi_rank.access(s,i) = ( 0 == comm_rank ) ? comm_size-1 : comm_rank-1;
                    printf("# %d EXPORT (%lu,%lu)->%d, axis: %lu\n",
                        comm_rank, s, i, mpi_rank.access(s,i), axis);
                }
                else if (is_leaving_domain == 1) { // -1 on y face
                    iy = (ny-1) + num_ghosts;
                }
                else if (is_leaving_domain == 2) { // -1 on z face
                    iz = (nz-1) + num_ghosts;
                }
                else if (is_leaving_domain == 3) { // 1 on x face
                    //ix = num_ghosts;
                    mpi_rank.access(s,i) = ( comm_size-1 == comm_rank ) ? 0 : comm_rank+1;
                    printf("# %d EXPORT (%lu,%lu)->%d, axis: %lu\n",
                        comm_rank, s, i, mpi_rank.access(s,i), axis);
                }
                else if (is_leaving_domain == 4) { // 1 on y face
                    iy = num_ghosts;
                }
                else if (is_leaving_domain == 5) { // 1 on z face
                    iz = num_ghosts;
                }
                /* int updated_ii = VOXEL(ix, iy, iz, */
                /*         nx, */
                /*         ny, */
                /*         nz, */
                /*         num_ghosts); */
            }

            /*         if ( Parameters::instance().BOUNDARY_TYPE == Boundary::Reflect) */
            /*         { */
            /*             // Hit a reflecting boundary condition.  Reflect the particle */
            /*             // momentum and remaining displacement and keep moving the */
            /*             // particle. */

            /*             //logger << "Reflecting " << s << " " << i << " on axis " << axis << std::endl; */

            /*             //(&(p->ux    ))[axis] = -(&(p->ux    ))[axis]; */
            /*             //(&(pm->dispx))[axis] = -(&(pm->dispx))[axis]; */
            /*             if (axis == 0) */
            /*             { */
            /*                 velocity_x.access(s, i) = -1.0f * velocity_x.access(s, i); */
            /*                 pm.dispx = -1.0f * s_dispx; */
            /*             } */
            /*             if (axis == 1) */
            /*             { */
            /*                 velocity_y.access(s, i) = -1.0f * velocity_y.access(s, i); */
            /*                 pm.dispy = -1.0f * s_dispy; */
            /*             } */
            /*             if (axis == 2) */
            /*             { */
            /*                 velocity_z.access(s, i) = -1.0f * velocity_z.access(s, i); */
            /*                 pm.dispz = -1.0f * s_dispz; */
            /*             } */
            /*             continue; */
            /*         } */
        }

        /*     // TODO: this nieghbor stuff can be removed by going to more simple */
        /*     // boundaries */
        /*     /\* */
        /*     if ( neighbor<g->rangel || neighbor>g->rangeh ) { */
        /*         // Cannot handle the boundary condition here.  Save the updated */
        /*         // particle position, face it hit and update the remaining */
        /*         // displacement in the particle mover. */
        /*         //p->i = 8*p->i + face; */
        /*         cell.access(s, i) = 8 * ii + face; */

        /*         return 1; // Return "mover still in use" */
        /*     } */
        /*     *\/ */
        /*     else { */

        /*     // Crossed into a normal voxel.  Update the voxel index, convert the */
        /*     // particle coordinate system and keep moving the particle. */

        /*     //p->i = neighbor - g->rangel; // Compute local index of neighbor */
        /*     //cell.access(s, i) = neighbor - g->rangel; */
        /*     // TODO: I still need to update the cell we're in */

        //1D only
        int updated_ii = ix+(nx+2)*(ny+2) + (nx+2);
        cell.access(s, i) = updated_ii;

        /* int updated_ii = VOXEL(ix, iy, iz, */
        /*         nx, */
        /*         ny, */
        /*         nz, */
        /*         num_ghosts); */

        /* cell.access(s, i) = updated_ii; */
        /*     //std::cout << "Moving from cell " << ii << " to " << updated_ii << std::endl; */
        /* } */

        /**/                         // Note: neighbor - g->rangel < 2^31 / 6
        //(&(p->dx))[axis] = -v0;      // Convert coordinate system
        // TODO: this conditional/branching could be better
        if (axis == 0) position_x.access(s, i) = -v0;
        if (axis == 1) position_y.access(s, i) = -v0;
        if (axis == 2) position_z.access(s, i) = -v0;

    }
    
    // Neighbor array would handle this better
    int ii = cell.access(s,i);
    int ix, iy, iz;
    RANK_TO_INDEX(ii, ix, iy, iz, nx+2*num_ghosts, ny+2*num_ghosts);
    if ( iy == 1 && iz == 1 ) { // NOTE: conditionals could be optimized
        //printf("(%lu,%lu) ix: %d\n", s, i, ix);
        if ( ix == 0 ) {
            ix == (nx-1) + num_ghosts;
            int updated_ii = ix+(nx+2)*(ny+2) + (nx+2);
            cell.access(s, i) = updated_ii;
        }
        if ( ix == nx-1+2*num_ghosts ) {
            ix == num_ghosts;
            int updated_ii = ix+(nx+2)*(ny+2) + (nx+2);
            cell.access(s, i) = updated_ii;
        }
    }

    return 0; // Return "mover not in use"
}

#endif // move_p
