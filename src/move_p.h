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

    if (ix < num_ghosts)
    {
        leaving = 0;
    }

    if (iy < num_ghosts)
    {
        leaving = 1;
    }

    if (iz < num_ghosts)
    {
        leaving = 2;
    }

    if (ix > (nx-1)+num_ghosts)
    {
        leaving = 3;
    }

    if (iy > (ny-1)+num_ghosts)
    {
        leaving = 4;
    }

    if (iz > (nz-1)+num_ghosts)
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
        const Boundary boundary,
        MPI_Comm mpi_comm
    )
{

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
    real_t v0, v1, v2, v3, v4, v5;
    size_t axis, face;
    size_t ix, iy, iz;
    /* //particle_t* p = p0 + pm->i; */
    /* //int index = pm->i; */

    //q = qsp * weight.access(s, i);

    for(;;)
    {

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
        v5 = q*s_dispx*s_dispy*s_dispz*(1./3.);

        int ii = cell.access(s, i);

        //a = (float *)(a0 + ii);

        //1D only
        //_asa(ii, accumulator_var::jx, 0) += q*s_dispx;
        //_asa(ii, accumulator_var::jx, 1) += 0.0;
        //_asa(ii, accumulator_var::jx, 2) += 0.0;
        //_asa(ii, accumulator_var::jx, 3) += 0.0;

        #define ACCUMULATE_J(X,Y,Z)                                         \
            v4  = q*s_disp##X;  /* v2 = q ux                            */  \
            v1  = v4*s_mid##Y;  /* v1 = q ux dy                         */  \
            v0  = v4-v1;        /* v0 = q ux (1-dy)                     */  \
            v1 += v4;           /* v1 = q ux (1+dy)                     */  \
            v4  = 1+s_mid##Z;   /* v4 = 1+dz                            */  \
            v2  = v0*v4;        /* v2 = q ux (1-dy)(1+dz)               */  \
            v3  = v1*v4;        /* v3 = q ux (1+dy)(1+dz)               */  \
            v4  = 1-s_mid##Z;   /* v4 = 1-dz                            */  \
            v0 *= v4;           /* v0 = q ux (1-dy)(1-dz)               */  \
            v1 *= v4;           /* v1 = q ux (1+dy)(1-dz)               */  \
            v0 += v5;           /* v0 = q ux [ (1-dy)(1-dz) + uy*uz/3 ] */  \
            v1 -= v5;           /* v1 = q ux [ (1+dy)(1-dz) - uy*uz/3 ] */  \
            v2 -= v5;           /* v2 = q ux [ (1-dy)(1+dz) - uy*uz/3 ] */  \
            v3 += v5;           /* v3 = q ux [ (1+dy)(1+dz) + uy*uz/3 ] */  \
            _asa(ii, accumulator_var::j##X, 0) += v0;                  \
            _asa(ii, accumulator_var::j##X, 1) += v1;                  \
            _asa(ii, accumulator_var::j##X, 2) += v2;                  \
            _asa(ii, accumulator_var::j##X, 3) += v3;                  

            ACCUMULATE_J(x,y,z);
            ACCUMULATE_J(y,z,x);
            ACCUMULATE_J(z,x,y);

        #undef ACCUMULATE_J

        // Compute the remaining particle displacment
        pm.dispx -= s_dispx;
        pm.dispy -= s_dispy;
        pm.dispz -= s_dispz;

        // Compute the new particle offset
        position_x.access(s, i) += s_dispx+s_dispx;
        position_y.access(s, i) += s_dispy+s_dispy;
        position_z.access(s, i) += s_dispz+s_dispz;

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

        RANK_TO_INDEX(ii, ix, iy, iz, nx+(2*num_ghosts), ny+(2*num_ghosts));

        if (face == 0) { ix--; }
        if (face == 1) { iy--; }
        if (face == 2) { iz--; }
        if (face == 3) { ix++; }
        if (face == 4) { iy++; }
        if (face == 5) { iz++; }

        //int is_leaving_domain = detect_leaving_domain(face, nx, ny, nz, ix, iy, iz, num_ghosts);
        //if (is_leaving_domain >= 0) {
        //    /*         if ( Parameters::instance().BOUNDARY_TYPE == Boundary::Reflect) */
        //    /*         { */
        //    /*             // Hit a reflecting boundary condition.  Reflect the particle */
        //    /*             // momentum and remaining displacement and keep moving the */
        //    /*             // particle. */

        //    /*             //logger << "Reflecting " << s << " " << i << " on axis " << axis << std::endl; */

        //    /*             //(&(p->ux    ))[axis] = -(&(p->ux    ))[axis]; */
        //    /*             //(&(pm->dispx))[axis] = -(&(pm->dispx))[axis]; */
        //    /*             if (axis == 0) */
        //    /*             { */
        //    /*                 velocity_x.access(s, i) = -1.0f * velocity_x.access(s, i); */
        //    /*                 pm.dispx = -1.0f * s_dispx; */
        //    /*             } */
        //    /*             if (axis == 1) */
        //    /*             { */
        //    /*                 velocity_y.access(s, i) = -1.0f * velocity_y.access(s, i); */
        //    /*                 pm.dispy = -1.0f * s_dispy; */
        //    /*             } */
        //    /*             if (axis == 2) */
        //    /*             { */
        //    /*                 velocity_z.access(s, i) = -1.0f * velocity_z.access(s, i); */
        //    /*                 pm.dispz = -1.0f * s_dispz; */
        //    /*             } */
        //    /*             continue; */
        //    /*         } */
        //}

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

        int updated_ii = VOXEL(ix, iy, iz,
                nx,
                ny,
                nz,
                num_ghosts);

        cell.access(s, i) = updated_ii;
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
    // TODO: currently assumes periodic boundary
    if ( detect_leaving_domain(face, nx, ny, nz, ix, iy, iz, num_ghosts) == -1 )
        return 0;
    // Get mpi info
    int dims[3];
    int wrap[3];
    int coords[3];
    MPI_Cart_get( mpi_comm, 3, dims, wrap, coords );
    if ( boundary == Boundary::Periodic)
    {
        RANK_TO_INDEX( ii, ix, iy, iz, nx+2*num_ghosts, ny+2*num_ghosts);
        int cx = coords[0];
        int cy = coords[1];
        int cz = coords[2];
        if ( ix < num_ghosts ) {
            cx = (ix + 1 - num_ghosts)/nx - 1 + coords[0];
            cx = ( cx < 0 ) ? cx + dims[0] : cx;
            ix = (nx-1) + num_ghosts; // TODO: enable mirror
        }
        else if ( ix > (nx-1)+num_ghosts ) {
            cx = (ix - nx + num_ghosts)/nx + 1 + coords[0];
            cx = ( cx > dims[0] ) ? cx - dims[0] : cx;
            ix = num_ghosts; // TODO: enable mirror
        }
        //if ( iy < num_ghosts ) {
        //    cy = (iy + 1 - num_ghosts)/ny - 1 + coords[1];
        //    cy = ( cy < 0 ) ? cy + dims[1] : cy;
        //}
        //else if ( iy > (ny-1)+num_ghosts ) {
        //    cy = (iy - ny + num_ghosts)/ny + 1 + coords[1];
        //    cy = ( cy > dims[1] ) ? cy - dims[1] : cy;
        //}
        //if ( iz < num_ghosts ) {
        //    cz = (iz - 1 + num_ghosts)/nz + 1 + coords[2];
        //    cz = ( cz < 0 ) ? cz + dims[2] : cz;
        //}
        //else if ( iz > (nz-1)+num_ghosts ) {
        //    cz = (iz - nz + num_ghosts)/nz + 1 + coords[2];
        //    cz = ( cz > dims[2] ) ? cz - dims[2] : cz;
        //}
        int dest_rank = -1;
        int c[3] = {cx, cy, cz};
        MPI_Cart_rank( mpi_comm, c, &dest_rank );
        mpi_rank.access(s,i) = dest_rank;
        int updated_ii = VOXEL(ix,iy,iz,nx,ny,nz,num_ghosts);
        cell.access(s, i) = updated_ii;
    }
    //if ( Parameters::instance().BOUNDARY_TYPE == Boundary::Reflect)
    //{
    //    // Hit a reflecting boundary condition.  Reflect the particle
    //    // momentum and remaining displacement and keep moving the particle.
    //    //logger << "Reflecting " << s << " " << i << " on axis " << axis << std::endl; */
    //    //(&(p->ux    ))[axis] = -(&(p->ux    ))[axis]; */
    //    //(&(pm->dispx))[axis] = -(&(pm->dispx))[axis]; */
    //    if (axis == 0)
    //    {
    //        velocity_x.access(s, i) = -1.0f * velocity_x.access(s, i);
    //        pm.dispx = -1.0f * s_dispx;
    //    }
    //    if (axis == 1)
    //    {
    //        velocity_y.access(s, i) = -1.0f * velocity_y.access(s, i);
    //        pm.dispy = -1.0f * s_dispy;
    //    }
    //    if (axis == 2)
    //    {
    //        velocity_z.access(s, i) = -1.0f * velocity_z.access(s, i);
    //        pm.dispz = -1.0f * s_dispz;
    //    }
    //    continue;
    //}
    /**/           //mirror( ii, nx, ny, nz, num_ghosts ); (accumulator.cpp)

    return 0; // Return "mover not in use"
}

#endif // move_p

