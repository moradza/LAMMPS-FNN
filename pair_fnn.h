/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(fnn,PairFNN)

#else

#ifndef LMP_PAIR_FNN_H
#define LMP_PAIR_FNN_H

#include "pair.h"
#include <cmath>
#include "iostream"
namespace LAMMPS_NS {

class PairFNN : public Pair {
 public:
  PairFNN(class LAMMPS *);
  virtual ~PairFNN();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  virtual double init_one(int, int);
  virtual void init_style();

  struct Network{
    double cut,cutsq;
    double fmax2, fmax3;
    int ielement,jelement,kelement;

    int num_layertwobody; // number of hidden layers of 2body network
    int *archtwobody; // topology of network of 2body
    char *functwobody; // non-linearity between layers of 2body 'n' = None, 't'= tanh, 's' = sigmoid
    double ***Wtwobody; // weights of 2body network
    double **btwobody;  // biases of 2body network
    int num_layerthreebody; // number of hidden layers of 3body network
    int *archthreebody; // topology of network of 3body
    char *functhreebody; // non-linearity between layers of 3body 'n' =None, 't' = tanh, 's' = sigmoid
    double ***Wthreebody; // weights of 3body network
    double **bthreebody; // biases of 3body network

    Network(){
	    fmax2= fmax3 = 20;
      int i, j, k;
      // two body network
      num_layertwobody = 1;
      archtwobody = new int [3];
      archtwobody[0] = archtwobody[1] = archtwobody[2] = 1;
      functwobody = new char [3];
      functwobody[0] = functwobody[1] = functwobody[2] = 'n';

      Wtwobody = new double **[3];
      for (i =0; i <2; i++){
           Wtwobody[i] = new double *[archtwobody[i+1]];
           for (j = 0; j < archtwobody[i+1]; j++)
               Wtwobody[i][j] = new double [archtwobody[i]];
       }

       for(i =0 ; i < 2; i++){
          for(j = 0; j < archtwobody[i+1]; j++){
            for(k =0 ; k < archtwobody[i]; k++)
               Wtwobody[i][j][k] = 0.0;
          }
       }

       btwobody = new double *[3];
       for (i = 0; i < 2; i++)
           btwobody[i] = new double [archtwobody[i+1]];

       for (i = 0; i < 2; i++){
           for (j = 0; j < archtwobody[i+1]; j++)
               btwobody[i][j] = 0.0;
       }
       // three body network
       num_layerthreebody = 1;

       archthreebody = new int [3];
       archthreebody[0] = 3; archthreebody[1] = 1; archthreebody[2] = 4;

       functhreebody = new char [3];
       functhreebody[0] = functhreebody[1] = functhreebody[2] = 'n';

       Wthreebody = new double **[3];
       for (i =0; i <2; i++){
           Wthreebody[i] = new double *[archthreebody[i+1]];
           for (j = 0; j < archthreebody[i+1]; j++)
               Wthreebody[i][j] = new double [archthreebody[i]];
       }

       for (i =0; i <2; i++){
           for (j = 0; j < archthreebody[i+1]; j++){
              for (k = 0; k < archthreebody[i]; k ++)
                   Wthreebody[i][j][k] = 0.0;
           }
       }

       bthreebody = new double *[3];
       for (i = 0; i < 2; i++)
           bthreebody[i] = new double [archthreebody[i+1]];

      for (i = 0; i < 2; i++){
        for (j =0; j < archthreebody[i+1]; j++)
          bthreebody[i][j] = 0.0;
      }

     }
  void multiply(double **A,  int *A_size, double *x, double *y){
    int nrow = A_size[0];
    int ncol = A_size[1];

    for (int row =0; row < nrow; row++)
        for (int col = 0; col < ncol; col++)
            y[col] += A[col][row] * x[row];
    }
  void add_bias(double *b, int b_size, double *x){
      for (int col = 0; col < b_size; col++) x[col] += b[col];

  }
  void apply_nonlinearity(char func, double *x, int x_size){
      for (int col =0 ; col < x_size; col++){
         if (func == 't')
             x[col] = (exp(x[col])-exp(-x[col])) / (exp(x[col])+exp(-x[col]));
         else if (func == 's')
             x[col] = 1.0/ (1.0+exp(-x[col]));
         else if (func == 'c')
             x[col] = 1.0/ (1.0+exp(-x[col])) + ((exp(x[col])-exp(-x[col])) / (exp(x[col])+exp(-x[col])));
         else if (func == 'n')
             x[col] = x[col];
       }
    }

  void change2body(int n_two, int *arch_two, char *func_two, double **b2_in, double ***W2_in, double f2_max){
       int i, j, k;
       fmax2 = f2_max;
       // clear two body objects
       //if (archtwobody) delete [] archtwobody;
       //if (functwobody) delete [] functwobody;
       //if (btwobody) delete [] btwobody;
       //if (Wtwobody) delete [] Wtwobody;


       // change size of two body objects and insert their values
       num_layertwobody = n_two;

       archtwobody = new int [num_layertwobody+2];
       for(i =0; i < num_layertwobody+2; i++) archtwobody[i] = arch_two[i];

       functwobody = new char [num_layertwobody+2];
       for (i =0; i < num_layertwobody+2; i++) functwobody[i] = func_two[i];

       btwobody = new double *[num_layertwobody+1];
       for (i = 0; i < num_layertwobody+1; i ++){
         btwobody[i] = new double [archtwobody[i+1]];
         for (j =0; j < archtwobody[i+1]; j++)
             btwobody[i][j] = b2_in[i][j];
       }

       Wtwobody = new double **[num_layertwobody+1];
       for (i =0; i < num_layertwobody+1; i++){
           Wtwobody[i] = new double *[archtwobody[i+1]];
           for (j =0; j < archtwobody[i+1] ; j++){
               Wtwobody[i][j] = new double [archtwobody[i]];
               for ( k = 0; k < archtwobody[i]; k++)
                     Wtwobody[i][j][k] = W2_in[i][j][k];
           }
       }
     }

   void change3body(int n_three, int *arch_three, char * func_three, double **b3_in, double ***W3_in, double f3_max){
          int i, j, k;
          fmax3 = f3_max;
          // clear three body objects
          //if (archthreebody) delete [] archthreebody;
          //if (functhreebody) delete [] functhreebody;
          //if (bthreebody) delete [] bthreebody;
          //if (Wthreebody) delete [] Wthreebody;

          // change size of three body objects and insert their values
          num_layerthreebody = n_three;

          archthreebody = new int [num_layerthreebody+2];
          for(i =0; i < num_layerthreebody+2; i++) archthreebody[i] = arch_three[i];

          functhreebody = new char [num_layerthreebody+2];
          for (i =0; i < num_layerthreebody+2; i++) functhreebody[i] = func_three[i];

          bthreebody = new double *[num_layerthreebody+1];
          for (i = 0; i < num_layerthreebody+1; i ++){
             bthreebody[i] = new double [archthreebody[i+1]];
             for (j =0; j < archthreebody[i+1]; j++)
                  bthreebody[i][j] = b3_in[i][j];
          }

          Wthreebody = new double **[num_layerthreebody+1];
          for (i =0; i < num_layerthreebody+1; i++){
              Wthreebody[i] = new double *[archthreebody[i+1]];
              for (j =0; j < archthreebody[i+1] ; j++){
                  Wthreebody[i][j] = new double [archthreebody[i]];
                  for ( k = 0; k < archthreebody[i]; k++)
                      Wthreebody[i][j][k]=W3_in[i][j][k];
              }
          }
   }

   Network(int n_two, int *arch_two, char *func_two, int n_three, int *arch_three, char * func_three){
         int i, j, k;
         num_layertwobody = n_two;

         archtwobody = new int [num_layertwobody+2];
         for(i =0; i < num_layertwobody+2; i++) archtwobody[i] = arch_two[i];

         functwobody = new char [num_layertwobody+2];
         for (i =0; i < num_layertwobody+2; i++) functwobody[i] = func_two[i];

         Wtwobody = new double **[num_layertwobody+1];
         for (i =0; i < num_layertwobody+1; i++){
             Wtwobody[i] = new double *[archtwobody[i+1]];
             for (j =0; j < archtwobody[i+1] ; j++){
                 Wtwobody[i][j] = new double [archtwobody[i]];
             }
         }

         for (i = 0; i < num_layertwobody +1 ; i++){
             for ( j = 0; j < archtwobody[i+1]; j++){
                 for ( k = 0; k < archtwobody[i]; k++) Wtwobody[i][j][k]=0.0;
             }
          }

         btwobody = new double *[num_layertwobody];
         for (i = 0; i < num_layertwobody; i ++) btwobody[i] = new double [archtwobody[i+1]];

        for (i = 0; i < num_layertwobody; i++){
            for (j =0; j < archtwobody[i+1]; j++)
                btwobody[i][j] = 0.0;
        }

         num_layerthreebody = n_three;
         archthreebody = new int [num_layerthreebody+2];
         for(i =0; i < num_layerthreebody+2; i++) archthreebody[i] = arch_three[i];

         functhreebody = new char [num_layerthreebody+2];
         for (i =0; i < num_layerthreebody+2; i++) functhreebody[i] = func_three[i];

         Wthreebody = new double **[num_layerthreebody+1];

         for (i =0; i < num_layerthreebody+1; i++){
             Wthreebody[i] = new double *[archthreebody[i+1]];
             for (j =0; j < archthreebody[i+1] ; j++){
                 Wthreebody[i][j] = new double [archthreebody[i]];
             }
         }

         for (i = 0; i < num_layerthreebody +1 ; i++){
             for (j = 0; j < archthreebody[i+1]; j++){
                 for ( k = 0; k < archthreebody[i]; k++) Wthreebody[i][j][k]=0.0;
             }
         }

         bthreebody = new double *[num_layerthreebody];
         for (i = 0; i < num_layerthreebody; i ++) bthreebody[i] = new double [archthreebody[i+1]];

         for (i = 0; i < num_layerthreebody; i++){
             for (j =0; j < archthreebody[i+1]; j++)
                 bthreebody[i][j] = 0.0;
         }
    }

    ~Network(void){

         if (Wtwobody){
           for (int i = 0; i < num_layertwobody+1; i++){
             for (int j = 0; j < archtwobody[i+1]; j++) delete [] Wtwobody[i][j];
             delete [] Wtwobody[i];
           }
           delete [] Wtwobody;
         }
         if (btwobody) {
           for (int i = 0; i < num_layertwobody+1; i++) delete [] btwobody[i];
           delete [] btwobody;
         }
         if (functwobody) delete [] functwobody; // non-linearity between layers of 2body 'n' = None, 't'= tanh, 's' = sigmoid
         if (archtwobody)  delete [] archtwobody ; // topology of network of 2body


         if (Wthreebody){
           for (int i = 0; i < num_layerthreebody+1; i++){
             for (int j = 0; j < archthreebody[i+1]; j++) delete [] Wthreebody[i][j];
             delete [] Wthreebody[i];
           }
           delete [] Wthreebody;
         }
         if (bthreebody) {
           for (int i = 0; i < num_layerthreebody+1; i++) delete [] bthreebody[i];
           delete [] bthreebody;
         }
         if (functhreebody) delete [] functhreebody;
         if (archthreebody) delete [] archthreebody;
      }
  };

 protected:
  double cutmax;                // max cutoff for all elements
  int nelements;                // # of unique elements
  char **elements;              // names of unique elements
  int ***elem2net;            // mapping from element triplets to parameters
  int *map;                     // mapping from atom types to elements
  int nnets;                  // # of stored network sets
  int maxnet;                 // max # of network sets
  Network *nets;                // network set for an I-J-K interaction
  int maxshort;                 // size of short neighbor list array
  int *neighshort;              // short neighbor list array

  virtual void allocate();
  void read_file(char *, char *, int, int *, int, int *);
  virtual void setup_params();
  void twobody(Network *, double, double &, int, double &);
  void threebody(Network *, double, double, double, double *, double *,
                 double *, double *, int, double &);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair style Stillinger-Weber requires atom IDs

This is a requirement to use the FNN potential.

E: Pair style Stillinger-Weber requires newton pair on

See the newton command.  This is a restriction to use the FNN
potential.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

E: Cannot open Stillinger-Weber potential file %s

The specified FNN potential file cannot be opened.  Check that the path
and name are correct.

E: Incorrect format in Stillinger-Weber potential file

Incorrect number of words per line in the potential file.

E: Illegal Stillinger-Weber parameter

One or more of the coefficients defined in the potential file is
invalid.

E: Potential file has duplicate entry

The potential file has more than one entry for the same element.

E: Potential file is missing an entry

The potential file does not have a needed entry.

*/
