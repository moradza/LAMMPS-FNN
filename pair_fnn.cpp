/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Aidan Thompson (SNL)
------------------------------------------------------------------------- */

#include "pair_fnn.h"
#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include <iostream>

using namespace LAMMPS_NS;

#define MAXLINE 10240
#define DELTA 4

/* ---------------------------------------------------------------------- */

PairFNN::PairFNN(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  eflag_global = 0;
  evflag = 0;
  vflag_fdotr = 0;

  nelements = 0;
  elements = NULL;
  nnets = maxnet = 0;
  nets = NULL;
  elem2net = NULL;
  map = NULL;

  maxshort = 10;
  neighshort = NULL;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairFNN::~PairFNN()
{
  if (copymode) return;

  if (elements)
    for (int i = 0; i < nelements; i++) delete [] elements[i];
  delete [] elements;
  memory->destroy(nets);
  memory->destroy(elem2net);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(neighshort);
    delete [] map;
  }
}

/* ---------------------------------------------------------------------- */

void PairFNN::compute(int eflag, int vflag)
{
  int i,j,k,ii,jj,kk,inum,jnum,jnumm1;
  int itype,jtype,ktype,ijparam,ijkparam;
  tagint itag,jtag;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,rsq1,rsq2, rsq3;
  double delr1[3],delr2[3], delr3[3], fj[3],fk[3];
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double fxtmp,fytmp,fztmp;

  // loop over full neighbor list of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itag = tag[i];
    itype = map[type[i]];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    fxtmp = fytmp = fztmp = 0.0;

    // two-body interactions, skip half of them

    jlist = firstneigh[i];
    jnum = numneigh[i];
    int numshort = 0;
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;

      jtype = map[type[j]];
      ijparam = elem2net[itype][jtype][jtype];
      if (rsq >= nets[ijparam].cutsq) {
        continue;
      } else {
        neighshort[numshort++] = j;
        if (numshort >= maxshort) {
          maxshort += maxshort/2;
          memory->grow(neighshort,maxshort,"pair:neighshort");
        }
      }

      jtag = tag[j];
      if (itag > jtag) {
        if ((itag+jtag) % 2 == 0) continue;
      } else if (itag < jtag) {
        if ((itag+jtag) % 2 == 1) continue;
      } else {
        if (x[j][2] < ztmp) continue;
        if (x[j][2] == ztmp && x[j][1] < ytmp) continue;
        if (x[j][2] == ztmp && x[j][1] == ytmp && x[j][0] < xtmp) continue;
      }
      twobody(&nets[ijparam],rsq,fpair,eflag,evdwl);
      fxtmp += delx*fpair;
      fytmp += dely*fpair;
      fztmp += delz*fpair;

      f[j][0] -= delx*fpair;
      f[j][1] -= dely*fpair;
      f[j][2] -= delz*fpair;
    }

    jnumm1 = numshort - 1;

    for (jj = 0; jj < jnumm1; jj++) {
      j = neighshort[jj];
      jtype = map[type[j]];
      delr1[0] = x[j][0] - xtmp;
      delr1[1] = x[j][1] - ytmp;
      delr1[2] = x[j][2] - ztmp;
      rsq1 = delr1[0]*delr1[0] + delr1[1]*delr1[1] + delr1[2]*delr1[2];

      double fjxtmp,fjytmp,fjztmp;
      fjxtmp = fjytmp = fjztmp = 0.0;

      for (kk = jj+1; kk < numshort; kk++) {
        k = neighshort[kk];
        ktype = map[type[k]];
        ijkparam = elem2net[itype][jtype][ktype];

        delr2[0] = x[k][0] - xtmp;
        delr2[1] = x[k][1] - ytmp;
        delr2[2] = x[k][2] - ztmp;
        rsq2 = delr2[0]*delr2[0] + delr2[1]*delr2[1] + delr2[2]*delr2[2];

        delr3[0] = x[k][0] - x[j][0];
        delr3[1] = x[k][1] - x[j][1];
        delr3[2] = x[k][2] - x[j][2];
        rsq3 = delr3[0]*delr3[0] + delr3[1]*delr3[1] + delr3[2]*delr3[2];
        threebody(&nets[ijkparam],rsq1,rsq2,rsq3, delr1,delr2,fj,fk,eflag,evdwl);
        fxtmp -= fj[0] + fk[0];
        fytmp -= fj[1] + fk[1];
        fztmp -= fj[2] + fk[2];
        fjxtmp += fj[0];
        fjytmp += fj[1];
        fjztmp += fj[2];
        f[k][0] += fk[0];
        f[k][1] += fk[1];
        f[k][2] += fk[2];

      }
      f[j][0] += fjxtmp;
      f[j][1] += fjytmp;
      f[j][2] += fjztmp;
    }
    f[i][0] += fxtmp;
    f[i][1] += fytmp;
    f[i][2] += fztmp;
  } 

}

/* ---------------------------------------------------------------------- */

void PairFNN::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(neighshort,maxshort,"pair:neighshort");
  map = new int[n+1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairFNN::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");
  cutmax = force->numeric(FLERR,arg[0]);
  if (cutmax < 0 )  error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairFNN::coeff(int narg, char **arg)
{
  int i,j,n;

  if (!allocated) allocate();
  
  // * * file2 n x_1 ... x_n file3 m y_1 ... y_m str_1 ... str_nelem
  if (narg < 9)
     error->all(FLERR,"Incorrect args for pair coefficients");

  // insure I,J args are * *
  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
       error->all(FLERR,"Incorrect args for pair coefficients");

  // args specific to neural network
  int num2bodylayers = force->inumeric(FLERR, arg[3]);
  int file3body_arg = 4+num2bodylayers;
  int num3bodylayers = force->inumeric(FLERR, arg[file3body_arg+1]);
  int elem_arg = num3bodylayers+file3body_arg+2;

  // topology of network
  int *arch2body = new int [num2bodylayers+2];
  arch2body[0] = arch2body[num2bodylayers+1] = 1;
  for (i=1; i < num2bodylayers+1; i++)
      arch2body[i] = force->inumeric(FLERR, arg[3+i]);
  
  int *arch3body = new int [num3bodylayers+2];
  arch3body[0] =3; arch3body[num3bodylayers+1] = 4; //  input (r1, r2, r3) output (f_1xe_12, f_2xe_13, f_3xe_12, f_4xe_13)
  for (i=1; i < num3bodylayers+1; i++){
      arch3body[i] = force->inumeric(FLERR, arg[file3body_arg+i+1]);
  }
  if ( narg != 6+atom->ntypes+num3bodylayers+num2bodylayers)
       error->all(FLERR,"Incorrect args for pair coefficients");

  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if NULL
  // nelements = # of unique elements
  // elements = list of element names

  if (elements) {
      for (i = 0; i < nelements; i++) delete [] elements[i];
      delete [] elements;
  }
  
  elements = new char*[atom->ntypes];
  for (i = 0; i < atom->ntypes; i++) elements[i] = NULL;
  
  nelements = 0;
  for (i = elem_arg; i < narg; i++) {
    if (strcmp(arg[i],"NULL") == 0) {
      map[i- elem_arg+1] = -1;
      continue;
    }
    for (j = 0; j < nelements; j++)
      if (strcmp(arg[i],elements[j]) == 0) break;
    map[i- elem_arg+1] = j;
    if (j == nelements) {
      n = strlen(arg[i]) + 1;
      elements[j] = new char[n];
      strcpy(elements[j],arg[i]);
      nelements++;
    }
  }
  
  // read potential file and initialize potential parameters

  read_file(arg[2], arg[file3body_arg], num2bodylayers, arch2body, num3bodylayers, arch3body);
  setup_params();

  // clear setflag since coeff() called once with I,J = * *
  n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements

  int count = 0;
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairFNN::init_style()
{
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style Stillinger-Weber requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style Stillinger-Weber requires newton pair on");

  // need a full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairFNN::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  return cutmax;
}

/* ---------------------------------------------------------------------- */

void PairFNN::read_file(char *file2, char *file3, int num2bodylayers, int *arch2body, int num3bodylayers, int *arch3body)
{
  int params_per_line3 = 3+num3bodylayers+2+1; //check
  for (int i = 0; i < num3bodylayers+1; i++){
    params_per_line3 += arch3body[i]*arch3body[i+1];
    params_per_line3 += arch3body[i+1];
  }
  int params_per_line2 = 2 + num2bodylayers+2+1; //check
  for (int i = 0; i < num2bodylayers+1; i++){
    params_per_line2 += arch2body[i]*arch2body[i+1];
    params_per_line2 += arch2body[i+1];
  }

  char **words2 = new char*[params_per_line2+1];
  char **words3 = new char*[params_per_line3+1];


  memory->sfree(nets);
  nets = NULL;
  nnets = maxnet = 0;

  double ***W3;
  double **b3;
  double ***W2;
  double **b2;

  // open file on proc 0

  FILE *fp3;
  if (comm->me == 0) {
    fp3 = force->open_potential(file3);
    if (fp3 == NULL) {
      char str[128];
      snprintf(str,128,"Cannot open Neural-Network potential file %s",file3);
      error->one(FLERR,str);
    }
  }

  int n3,nwords3,ielement,jelement,kelement;
  int n2, nwords2;
  int cur_net =0;
  char line3[MAXLINE], *ptr3;
  char line2[MAXLINE], *ptr2;
  int eof3 = 0;
  int eof2 = 0;
  double fmax;

  while (1) {
    if (comm->me == 0) {
      ptr3 = fgets(line3, MAXLINE,fp3);
      if (ptr3 == NULL ) {
        eof3 = 1;
        fclose(fp3);
      } else n3 = strlen(line3) + 1;
    }

    MPI_Bcast(&eof3,1,MPI_INT,0,world);
    if (eof3) break;
    MPI_Bcast(&n3,1,MPI_INT,0,world);
    MPI_Bcast(line3,n3,MPI_CHAR,0,world);
    // strip comment, skip line if blank

    if ((ptr3 = strchr(line3,'#'))) *ptr3 = '\0';
    nwords3 = atom->count_words(line3);
    if (nwords3 == 0) continue;
    // concatenate additional lines until have params_per_line words

    while (nwords3 < params_per_line3) {
      n3 = strlen(line3);
      if (comm->me == 0) {
        ptr3 = fgets(&line3[n3],MAXLINE-n3,fp3);
        if (ptr3 == NULL) {
          eof3 = 1;
          fclose(fp3);
        } else n3 = strlen(line3) + 1;
      }
      MPI_Bcast(&eof3,1,MPI_INT,0,world);
      if (eof3) break;
      MPI_Bcast(&n3,1,MPI_INT,0,world);
      MPI_Bcast(line3,n3,MPI_CHAR,0,world);
      if ((ptr3 = strchr(line3,'#'))) *ptr3 = '\0';
      nwords3 = atom->count_words(line3);
    }

    if (nwords3 != params_per_line3)
       error->all(FLERR,"Incorrect format in Neural-Network potential file");
    // words = ptrs to all words in line

    nwords3 = 0;
    words3[nwords3++] = strtok(line3," \t\n\r\f");

    while ((words3[nwords3++] = strtok(NULL," \t\n\r\f"))) continue;
    // ielement,jelement,kelement = 1st args
    // if all 3 args are in element list, then parse this line
    // else skip to next entry in file
    for (ielement = 0; ielement < nelements; ielement++)
      if (strcmp(words3[0],elements[ielement]) == 0) break;
    if (ielement == nelements) continue;
    for (jelement = 0; jelement < nelements; jelement++)
      if (strcmp(words3[1],elements[jelement]) == 0) break;
    if (jelement == nelements) continue;
    for (kelement = 0; kelement < nelements; kelement++)
      if (strcmp(words3[2],elements[kelement]) == 0) break;
    if (kelement == nelements) continue;

    // load up parameter settings and error check their values
    if (nnets == maxnet) {
      maxnet += DELTA;
      nets = (Network *) memory->srealloc(nets,maxnet*sizeof(Network),
                                          "pair:nets");
    }

    nets[nnets].ielement = ielement;
    nets[nnets].jelement = jelement;
    nets[nnets].kelement = kelement;

    char *  func_three = new char [num3bodylayers+2];
    for (int k=3; k< 3+num3bodylayers+2; k++)
        func_three[k-3] = words3[k][0];
    // add a cnt for reading weights and biases
    int cnt3 = 3+num3bodylayers+2;

    b3 = new double *[num3bodylayers+1];
    W3 = new double **[num3bodylayers+1];


    // initialization of W2, W3, b2, b3
    for (int layer = 0 ; layer < num3bodylayers+1; layer++){
      b3[layer] = new double [arch3body[layer+1]];
      W3[layer] = new double *[arch3body[layer+1]];
      for(int i = 0 ; i < arch3body[layer+1]; i++){
        W3[layer][i] = new double [arch3body[layer]];
        b3[layer][i] = 0.0;
        for (int j = 0; j <  arch3body[layer]; j++)
          W3[layer][i][j] = 0.0;
      }
    }
    for(int layer=0; layer < num3bodylayers+1; layer++){
      for (int i=0; i< arch3body[layer+1]; i++)
        for(int j=0; j < arch3body[layer]; j++){
          W3[layer][i][j] = atof(words3[cnt3]);
          cnt3 ++;
        }
      for (int i=0; i< arch3body[layer+1]; i++){
        b3[layer][i] = atof(words3[cnt3]);
        cnt3 ++;
      }
    }
    fmax = atof(words3[cnt3]);
    nets[nnets].change3body(num3bodylayers,arch3body,func_three,b3,W3, fmax);
    // delete temporary pointers
    delete [] func_three;

    for (int layer = 0 ; layer < num3bodylayers+1; layer++){
      delete [] b3[layer];
      for(int i = 0 ; i < arch3body[layer+1]; i++)
        delete [] W3[layer][i];
      delete [] W3[layer];
    }
    delete [] W3;
    delete [] b3;

    // next network
    nnets ++;
  }

  FILE *fp2;
  if (comm->me == 0) {
    fp2 = force->open_potential(file2);
    if (fp2 == NULL) {
      char str[128];
      snprintf(str,128,"Cannot open Neural-Network potential file %s",file2);
      error->one(FLERR,str);
    }
  }

  // read each set of params from potential file
  // one set of params can span multiple lines
  // store params if all 3 element tags are in element list

  while (1) {
    if (comm->me == 0) {
      ptr2 = fgets(line2,MAXLINE,fp2);
      if (ptr2 == NULL ) {
        eof2 = 1;
        fclose(fp2);
      } else n2 = strlen(line2) + 1;
     }
    MPI_Bcast(&eof2,1,MPI_INT,0,world);
    if (eof2) break;
    MPI_Bcast(&n2,1,MPI_INT,0,world);
    MPI_Bcast(line2,n2,MPI_CHAR,0,world);

    // strip comment, skip line if blank
    if ((ptr2 = strchr(line2,'#'))) *ptr2 = '\0';
    nwords2 = atom->count_words(line2);
    if (nwords2 == 0) continue;

    // concatenate additional lines until have params_per_line words
    while (nwords2 < params_per_line2) {
      n2 = strlen(line2);
      if (comm->me == 0) {
        ptr2 = fgets(&line2[n2],MAXLINE-n2,fp2);
        if (ptr2 == NULL) {
          eof2 = 1;
          fclose(fp2);
        } else n2 = strlen(line2) + 1;
      }
      MPI_Bcast(&eof2,1,MPI_INT,0,world);
      if (eof2) break;
      MPI_Bcast(&n2,1,MPI_INT,0,world);
      MPI_Bcast(line2,n2,MPI_CHAR,0,world);
      if ((ptr2 = strchr(line2,'#'))) *ptr2 = '\0';
      nwords2 = atom->count_words(line2);
    }
    
    if (nwords2 != params_per_line2)
       error->all(FLERR,"Incorrect format in Neural-Network potential file");
    // words = ptrs to all words in line
    nwords2 = 0;
    words2[nwords2++] = strtok(line2," \t\n\r\f");
    while ((words2[nwords2++] = strtok(NULL," \t\n\r\f"))) continue;
    // ielement,jelement,kelement = 1st args
    // if all 3 args are in element list, then parse this line
    // else skip to next entry in file
    
    for (ielement = 0; ielement < nelements; ielement++)
        if (strcmp(words2[0],elements[ielement]) == 0) break;
    if (ielement == nelements) continue;
    for (jelement = 0; jelement < nelements; jelement++)
        if (strcmp(words2[1],elements[jelement]) == 0) break;
    if (jelement == nelements) continue;
    while (!(nets[cur_net].ielement == ielement && nets[cur_net].jelement == jelement)){
        cur_net++;
    }
    // load up parameter settings and error check their values
    char * func_two = new char [num2bodylayers+2];
    for (int k=3; k< 3+num2bodylayers+2; k++) //check if we need 3 or 2 to start enumerating
        func_two[k-3] = words2[k][0];
    // add a cnt for reading weights and biases
    int cnt2 = 3+num2bodylayers+1;

    b2 = new double *[num2bodylayers+1];
    W2 = new double **[num2bodylayers+1];

    for (int layer = 0 ; layer < num2bodylayers+1; layer++){
      b2[layer] = new double [arch2body[layer+1]];
      W2[layer] = new double *[arch2body[layer+1]];
      for(int i = 0 ; i < arch2body[layer+1]; i++){
        W2[layer][i] = new double [arch2body[layer]];
        b2[layer][i] = 0.0;
        for (int j = 0; j <  arch2body[layer]; j++)
          W2[layer][i][j] = 0.0;
      }
    }

    for(int layer=0; layer < num2bodylayers+1; layer++){
      for (int i=0; i< arch2body[layer+1]; i++)
        for(int j=0; j < arch2body[layer]; j++){
          W2[layer][i][j] = atof(words2[cnt2]);
          cnt2 ++;
      }
      for (int i=0; i< arch2body[layer+1]; i++){
        b2[layer][i] = atof(words2[cnt2]);
        cnt2 ++;
      }
    }
    fmax = atof(words2[cnt2]);
    nets[cur_net].change2body(num2bodylayers,arch2body,func_two,b2,W2, fmax);

    delete [] func_two;

    for (int layer = 0 ; layer < num2bodylayers+1; layer++){
      delete [] b2[layer];
      for(int i = 0 ; i < arch2body[layer+1]; i++)
        delete [] W2[layer][i];
      delete [] W2[layer];
    }
    delete [] W2;
    delete [] b2;
  }

  delete [] words2;
  delete [] words3;
}

/* ---------------------------------------------------------------------- */

void PairFNN::setup_params()
{
  int i,j,k,m,n;
  double rtmp;

  // set elem2param for all triplet combinations
  // must be a single exact match to lines read from file
  // do not allow for ACB in place of ABC

  memory->destroy(elem2net);
  memory->create(elem2net,nelements,nelements,nelements,"pair:elem2net");

  for (i = 0; i < nelements; i++)
    for (j = 0; j < nelements; j++)
      for (k = 0; k < nelements; k++) {
        n = -1;
        for (m = 0; m < nnets; m++) {
          if (i == nets[m].ielement && j == nets[m].jelement &&
              k == nets[m].kelement) {
            if (n >= 0) error->all(FLERR,"Potential file has duplicate entry");
            n = m;
          }
        }
        if (n < 0) error->all(FLERR,"Potential file is missing an entry");
        elem2net[i][j][k] = n;
      }

  // compute parameter values derived from inputs

  // set cutsq using shortcut to reduce neighbor list for accelerated
  // calculations. cut must remain unchanged as it is a potential parameter
  // (cut = a*sigma)

  for (m = 0; m < nnets; m++) {
    nets[m].cut =  cutmax;
    nets[m].cutsq = cutmax*cutmax;
  }
}

/* ---------------------------------------------------------------------- */

void PairFNN::twobody(Network *net, double rsq, double &fforce,
                     int eflag, double &eng)
{
  double r;
  int layer;
  r = sqrt(rsq);
  double *L_k, *L_kp;
  L_k = new double [1];
  L_kp = new double [net->archtwobody[1]];
  for (int i = 0 ; i < net->archtwobody[1]; i++) L_kp[i] = 0.0;

  L_k[0] = -(r-cutmax)/cutmax;
  int *A_size = new int [2];
  for (layer = 0 ; layer < net->num_layertwobody+1; layer ++ ){
    A_size[0] = net->archtwobody[layer]; A_size[1] = net->archtwobody[layer+1];

    net->multiply(net->Wtwobody[layer], A_size, L_k, L_kp);
    net->add_bias(net->btwobody[layer], A_size[1], L_kp);
    net->apply_nonlinearity(net->functwobody[layer], L_kp, A_size[1]);

    delete [] L_k;
    L_k = new double [net->archtwobody[layer+1]];
    for (int i = 0; i <net->archtwobody[layer+1]; i++) {
	    L_k[i] = L_kp[i];
    }
    delete [] L_kp;

    if (layer != net->num_layertwobody){
       L_kp = new double [net->archtwobody[layer+2]];
       for (int i = 0 ; i < net->archtwobody[layer+2]; i++) L_kp[i] = 0.0;
    } else{
       L_kp = new double [net->archtwobody[layer+1]];
       for (int i = 0 ; i < net->archtwobody[layer+1]; i++) L_kp[i] = 0.0;
    }
  }
  fforce = L_k[0]*net->fmax2;
  delete [] L_k;
  delete [] L_kp;
  delete [] A_size;
}

/* ---------------------------------------------------------------------- */

void PairFNN::threebody(Network *net,
                       double rsq1, double rsq2, double rsq3,
                       double *delr1, double *delr2,
                       double *fj, double *fk, int eflag, double &eng)
{
  double r1, r2, r3;
  r1 = sqrt(rsq1);
  r2 = sqrt(rsq2);
  r3 = sqrt(rsq3);

  double *L_k, *L_kp;
  L_k = new double [3];
  L_k[0] = -(r1-cutmax)/cutmax;
  L_k[1] = -(r2-cutmax)/cutmax;
  L_k[2] = -(r3-cutmax)/cutmax;

  int *A_size = new int [2];
  int b_size;
  for (int layer = 0 ; layer < net->num_layerthreebody+1; layer++){
    // check A if col and row should be changed!
    A_size[0] = net->archthreebody[layer]; A_size[1] = net->archthreebody[layer+1];
    b_size = net->archthreebody[layer+1];
    L_kp = new double [net->archthreebody[layer+1]];
    for (int i = 0 ; i < net->archthreebody[layer+1]; i++) L_kp[i] = 0.0;

    net->multiply(net->Wthreebody[layer], A_size, L_k, L_kp );
    net->add_bias(net->bthreebody[layer], b_size, L_kp);
    net->apply_nonlinearity(net->functhreebody[layer],L_kp, b_size);
    
    
    delete [] L_k;
    L_k = new double [b_size];
    for (int i = 0; i < b_size; i++) {
            L_k[i] = L_kp[i];
    }
    delete [] L_kp;
  }
  double er1[3], er2[3];
  er1[0] = delr1[0]/r1; er1[1] = delr1[1]/r1; er1[2] = delr1[2]/r1;
  er2[0] = delr2[0]/r2; er2[1] = delr2[1]/r2; er2[2] = delr2[2]/r2;

  fj[0] = er1[0]*L_k[0] + er2[0]*L_k[1];
  fj[1] = er1[1]*L_k[0] + er2[1]*L_k[1];
  fj[2] = er1[2]*L_k[0] + er2[2]*L_k[1];

  fj[0] = fj[0] * net->fmax3;
  fj[1] = fj[1] * net->fmax3;
  fj[2] = fj[2] * net->fmax3;

  fk[0] = er1[0]*L_k[2] + er2[0]*L_k[3];
  fk[1] = er1[1]*L_k[2] + er2[1]*L_k[3];
  fk[2] = er1[2]*L_k[2] + er2[2]*L_k[3];

  fk[0] = fk[0] * net->fmax3;
  fk[1] = fk[1] * net->fmax3;
  fk[2] = fk[2] * net->fmax3;

  delete [] L_k;
  delete [] A_size;
}
