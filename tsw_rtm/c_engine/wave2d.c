#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

/*
 * High-Performance 2D Wave Propagation and Reverse Time Migration Engine
 * Supports:
 * 1. 2D Elastic Staggered-Grid Finite Difference (Levander 1988) with Free Surface
 * 2. 2D Acoustic Wave Equation with Free-Surface and Sponge Boundaries
 * 3. On-the-fly Zero-Lag Cross-Correlation Imaging Condition Accumulator
 */

/* ========================================================================= */
/*                          1. ACOUSTIC 2D SOLVER                            */
/* ========================================================================= */

void acoustic_2d_step(
    const float* p_curr,
    const float* p_prev,
    float* p_next,
    const float* vel,
    const float* sponge,
    int nx,
    int nz,
    float dx,
    float dz,
    float dt,
    int free_surface_top
) {
    float idx2 = 1.0f / (dx * dx);
    float idz2 = 1.0f / (dz * dz);
    float dt2 = dt * dt;

    // 4th order coefficients
    float c0 = -2.5f;
    float c1 = 4.0f / 3.0f;
    float c2 = -1.0f / 12.0f;

    int z_start = free_surface_top ? 2 : 2;
    int z_end = nz - 2;

    #pragma omp parallel for schedule(static)
    for (int iz = z_start; iz < z_end; iz++) {
        int row = iz * nx;
        for (int ix = 2; ix < nx - 2; ix++) {
            int idx = row + ix;

            // 4th order laplacian in x
            float d2x = (c2 * (p_curr[idx - 2] + p_curr[idx + 2]) +
                         c1 * (p_curr[idx - 1] + p_curr[idx + 1]) +
                         c0 * p_curr[idx]) * idx2;

            // 4th order laplacian in z
            float d2z = (c2 * (p_curr[idx - 2 * nx] + p_curr[idx + 2 * nx]) +
                         c1 * (p_curr[idx - nx] + p_curr[idx + nx]) +
                         c0 * p_curr[idx]) * idz2;

            float v = vel[idx];
            float v2_dt2 = v * v * dt2;

            float p_new = 2.0f * p_curr[idx] - p_prev[idx] + v2_dt2 * (d2x + d2z);

            // Apply sponge absorbing boundary
            if (sponge) {
                p_new *= sponge[idx];
            }

            p_next[idx] = p_new;
        }
    }

    // Free surface boundary condition at top if requested: p(x, 0) = 0 or mirror
    if (free_surface_top) {
        #pragma omp parallel for schedule(static)
        for (int ix = 0; ix < nx; ix++) {
            p_next[0 * nx + ix] = 0.0f;
            p_next[1 * nx + ix] = -p_next[2 * nx + ix]; // Anti-symmetric reflection
        }
    }
}

/* ========================================================================= */
/*                          2. ELASTIC 2D SOLVER                             */
/* ========================================================================= */
/*
 * Velocity-stress formulation on standard staggered grid (Levander, 1988)
 * vx: at (ix + 0.5, iz)
 * vz: at (ix, iz + 0.5)
 * sxx, szz: at (ix, iz)
 * sxz: at (ix + 0.5, iz + 0.5)
 */

void elastic_2d_step(
    float* vx,
    float* vz,
    float* sxx,
    float* szz,
    float* sxz,
    const float* lambda,
    const float* mu,
    const float* buo_x, // 1 / rho_x
    const float* buo_z, // 1 / rho_z
    const float* sponge,
    int nx,
    int nz,
    float dx,
    float dz,
    float dt,
    int free_surface_top
) {
    float idx = 1.0f / dx;
    float idz = 1.0f / dz;

    // 4th order staggered difference coefficients
    float c1 = 9.0f / 8.0f;
    float c2 = -1.0f / 24.0f;

    // --- 1. Update velocities from stress derivatives ---
    int z_start_v = free_surface_top ? 1 : 2;

    #pragma omp parallel for schedule(static)
    for (int iz = z_start_v; iz < nz - 2; iz++) {
        int row = iz * nx;
        for (int ix = 2; ix < nx - 2; ix++) {
            int idx_cell = row + ix;

            // d(sxx)/dx at (ix + 0.5, iz)
            float dsxx_dx = (c1 * (sxx[idx_cell + 1] - sxx[idx_cell]) +
                             c2 * (sxx[idx_cell + 2] - sxx[idx_cell - 1])) * idx;

            // d(sxz)/dz at (ix + 0.5, iz)
            float dsxz_dz = (c1 * (sxz[idx_cell] - sxz[idx_cell - nx]) +
                             c2 * (sxz[idx_cell + nx] - sxz[idx_cell - 2 * nx])) * idz;

            vx[idx_cell] += dt * buo_x[idx_cell] * (dsxx_dx + dsxz_dz);

            // d(sxz)/dx at (ix, iz + 0.5)
            float dsxz_dx = (c1 * (sxz[idx_cell] - sxz[idx_cell - 1]) +
                             c2 * (sxz[idx_cell + 1] - sxz[idx_cell - 2])) * idx;

            // d(szz)/dz at (ix, iz + 0.5)
            float dszz_dz = (c1 * (szz[idx_cell + nx] - szz[idx_cell]) +
                             c2 * (szz[idx_cell + 2 * nx] - szz[idx_cell - nx])) * idz;

            vz[idx_cell] += dt * buo_z[idx_cell] * (dsxz_dx + dszz_dz);

            if (sponge) {
                float sp = sponge[idx_cell];
                vx[idx_cell] *= sp;
                vz[idx_cell] *= sp;
            }
        }
    }

    // --- 2. Update stresses from velocity derivatives ---
    #pragma omp parallel for schedule(static)
    for (int iz = z_start_v; iz < nz - 2; iz++) {
        int row = iz * nx;
        for (int ix = 2; ix < nx - 2; ix++) {
            int idx_cell = row + ix;

            // dvx/dx at (ix, iz)
            float dvx_dx = (c1 * (vx[idx_cell] - vx[idx_cell - 1]) +
                            c2 * (vx[idx_cell + 1] - vx[idx_cell - 2])) * idx;

            // dvz/dz at (ix, iz)
            float dvz_dz = (c1 * (vz[idx_cell] - vz[idx_cell - nx]) +
                            c2 * (vz[idx_cell + nx] - vz[idx_cell - 2 * nx])) * idz;

            float lam = lambda[idx_cell];
            float m = mu[idx_cell];

            sxx[idx_cell] += dt * ((lam + 2.0f * m) * dvx_dx + lam * dvz_dz);
            szz[idx_cell] += dt * (lam * dvx_dx + (lam + 2.0f * m) * dvz_dz);

            // dvz/dx at (ix + 0.5, iz + 0.5)
            float dvz_dx = (c1 * (vz[idx_cell + 1] - vz[idx_cell]) +
                            c2 * (vz[idx_cell + 2] - vz[idx_cell - 1])) * idx;

            // dvx/dz at (ix + 0.5, iz + 0.5)
            float dvx_dz = (c1 * (vx[idx_cell + nx] - vx[idx_cell]) +
                            c2 * (vx[idx_cell + 2 * nx] - vx[idx_cell - nx])) * idz;

            sxz[idx_cell] += dt * m * (dvz_dx + dvx_dz);

            if (sponge) {
                float sp = sponge[idx_cell];
                sxx[idx_cell] *= sp;
                szz[idx_cell] *= sp;
                sxz[idx_cell] *= sp;
            }
        }
    }

    // Free surface boundary condition at z = 0 (Levander 1988)
    if (free_surface_top) {
        #pragma omp parallel for schedule(static)
        for (int ix = 0; ix < nx; ix++) {
            szz[0 * nx + ix] = 0.0f;
            sxz[0 * nx + ix] = 0.0f;
            // sxx on surface determined by free surface relation: sxx = 4 * mu * (lambda+mu)/(lambda+2*mu) * dvx/dx
            float lam = lambda[ix];
            float m = mu[ix];
            if (ix >= 2 && ix < nx - 2) {
                float dvx_dx = (c1 * (vx[ix] - vx[ix - 1]) + c2 * (vx[ix + 1] - vx[ix - 2])) * idx;
                sxx[ix] = 4.0f * m * (lam + m) / (lam + 2.0f * m + 1e-8f) * dvx_dx;
            }
        }
    }
}

/* ========================================================================= */
/*                   3. ZERO-LAG CORRELATION IMAGING CONDITION               */
/* ========================================================================= */

void correlate_imaging_condition(
    const float* src_wavefield,
    const float* rec_wavefield,
    float* image,
    int nx,
    int nz,
    float dt
) {
    int n = nx * nz;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        image[i] += src_wavefield[i] * rec_wavefield[i] * dt;
    }
}

/* ========================================================================= */
/*                          4. SPONGE BOUNDARY BUILDER                       */
/* ========================================================================= */

void build_cerjan_sponge(
    float* sponge,
    int nx,
    int nz,
    int n_sponge,
    float alpha,
    int free_surface_top
) {
    for (int iz = 0; iz < nz; iz++) {
        for (int ix = 0; ix < nx; ix++) {
            float dist = 0.0f;
            float d_left = (float)(n_sponge - ix);
            float d_right = (float)(ix - (nx - 1 - n_sponge));
            float d_bottom = (float)(iz - (nz - 1 - n_sponge));
            float d_top = free_surface_top ? -1.0f : (float)(n_sponge - iz);

            if (d_left > dist) dist = d_left;
            if (d_right > dist) dist = d_right;
            if (d_bottom > dist) dist = d_bottom;
            if (!free_surface_top && d_top > dist) dist = d_top;

            if (dist > 0.0f) {
                sponge[iz * nx + ix] = expf(-alpha * dist * dist);
            } else {
                sponge[iz * nx + ix] = 1.0f;
            }
        }
    }
}
