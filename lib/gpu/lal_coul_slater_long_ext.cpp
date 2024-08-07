/***************************************************************************
                           coul_slater_long_ext.cpp
                           ------------------------
                           Trung Nguyen (U Chicago)

  Functions for LAMMPS access to coul/slater/long acceleration routines.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 __________________________________________________________________________

    begin                : September 2023
    email                : ndactrung@gmail.com
 ***************************************************************************/

#include <iostream>
#include <cassert>
#include <cmath>

#include "lal_coul_slater_long.h"

using namespace std;
using namespace LAMMPS_AL;

static CoulSlaterLong<PRECISION,ACC_PRECISION> CSLMF;

// ---------------------------------------------------------------------------
// Allocate memory on host and device and copy constants to device
// ---------------------------------------------------------------------------
int csl_gpu_init(const int ntypes, double **host_scale,
                const int inum, const int nall, const int max_nbors,
                const int maxspecial, const double cell_size, int &gpu_mode,
                FILE *screen, double host_cut_coulsq, double *host_special_coul,
                const double qqrd2e, const double g_ewald, const double lamda) {
  CSLMF.clear();
  gpu_mode=CSLMF.device->gpu_mode();
  double gpu_split=CSLMF.device->particle_split();
  int first_gpu=CSLMF.device->first_device();
  int last_gpu=CSLMF.device->last_device();
  int world_me=CSLMF.device->world_me();
  int gpu_rank=CSLMF.device->gpu_rank();
  int procs_per_gpu=CSLMF.device->procs_per_gpu();

  CSLMF.device->init_message(screen,"coul/slater/long",first_gpu,last_gpu);

  bool message=false;
  if (CSLMF.device->replica_me()==0 && screen)
    message=true;

  if (message) {
    fprintf(screen,"Initializing Device and compiling on process 0...");
    fflush(screen);
  }

  int init_ok=0;
  if (world_me==0)
    init_ok=CSLMF.init(ntypes, host_scale, inum, nall, max_nbors, maxspecial,
                      cell_size, gpu_split, screen, host_cut_coulsq,
                      host_special_coul, qqrd2e, g_ewald, lamda);

  CSLMF.device->world_barrier();
  if (message)
    fprintf(screen,"Done.\n");

  for (int i=0; i<procs_per_gpu; i++) {
    if (message) {
      if (last_gpu-first_gpu==0)
        fprintf(screen,"Initializing Device %d on core %d...",first_gpu,i);
      else
        fprintf(screen,"Initializing Devices %d-%d on core %d...",first_gpu,
                last_gpu,i);
      fflush(screen);
    }
    if (gpu_rank==i && world_me!=0)
      init_ok=CSLMF.init(ntypes, host_scale, inum, nall, max_nbors, maxspecial,
                        cell_size, gpu_split, screen, host_cut_coulsq,
                        host_special_coul, qqrd2e, g_ewald, lamda);

    CSLMF.device->serialize_init();
    if (message)
      fprintf(screen,"Done.\n");
  }
  if (message)
    fprintf(screen,"\n");

  if (init_ok==0)
    CSLMF.estimate_gpu_overhead();
  return init_ok;
}

// ---------------------------------------------------------------------------
// Copy updated coeffs from host to device
// ---------------------------------------------------------------------------
void csl_gpu_reinit(const int ntypes, double **host_scale) {
  int world_me=CSLMF.device->world_me();
  int gpu_rank=CSLMF.device->gpu_rank();
  int procs_per_gpu=CSLMF.device->procs_per_gpu();

  if (world_me==0)
    CSLMF.reinit(ntypes, host_scale);

  CSLMF.device->world_barrier();

  for (int i=0; i<procs_per_gpu; i++) {
    if (gpu_rank==i && world_me!=0)
      CSLMF.reinit(ntypes, host_scale);

    CSLMF.device->serialize_init();
  }
}

void csl_gpu_clear() {
  CSLMF.clear();
}

int** csl_gpu_compute_n(const int ago, const int inum_full,
                       const int nall, double **host_x, int *host_type,
                       double *sublo, double *subhi, tagint *tag, int **nspecial,
                       tagint **special, const bool eflag, const bool vflag,
                       const bool eatom, const bool vatom, int &host_start,
                       int **ilist, int **jnum,  const double cpu_time,
                       bool &success, double *host_q, double *boxlo,
                       double *prd) {
  return CSLMF.compute(ago, inum_full, nall, host_x, host_type, sublo,
                      subhi, tag, nspecial, special, eflag, vflag, eatom,
                      vatom, host_start, ilist, jnum, cpu_time, success,
                      host_q, boxlo, prd);
}

void csl_gpu_compute(const int ago, const int inum_full, const int nall,
                    double **host_x, int *host_type, int *ilist, int *numj,
                    int **firstneigh, const bool eflag, const bool vflag,
                    const bool eatom, const bool vatom, int &host_start,
                    const double cpu_time, bool &success, double *host_q,
                    const int nlocal, double *boxlo, double *prd) {
  CSLMF.compute(ago,inum_full,nall,host_x,host_type,ilist,numj,
               firstneigh,eflag,vflag,eatom,vatom,host_start,cpu_time,success,
               host_q,nlocal,boxlo,prd);
}

double csl_gpu_bytes() {
  return CSLMF.host_memory_usage();
}


