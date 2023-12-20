#include <sys/stat.h>
#include <iostream>
#include <complex>
#include <cmath>
#include <cstring>

// contour library headers
#include "cntr/cntr.hpp"
#include "cntr/utils/read_inputfile.hpp"
#include "omp.h"
// #include <cntr/cntr_function_decl.hpp>
// #include <src/Core/util/Memory.h>

// local headers to include
// #include "formats.hpp"
void print_line_minus(int width) {
  std::cout << std::string(width, '-') << std::endl;
}

void print_line_plus(int width) {
  std::cout << std::string(width, '+') << std::endl;
}

void print_line_equal(int width) {
  std::cout << std::string(width, '=') << std::endl;
}

void print_line_star(int width) {
  std::cout << std::string(width, '*') << std::endl;
}

void print_line_dot(int width) {
  std::cout << std::string(width, '.') << std::endl;
}



using namespace std;

#define CFUNC cntr::function<double>
#define GREEN cntr::herm_matrix<double>
#define GREEN_TSTP cntr::herm_matrix_timestep<double>
#define CPLX complex<double>
#define THREAD_NUM 12



//==============================================================================
//         main program
//==============================================================================
int main(int argc,char *argv[]){

const double dlt = 0.2;
const double mu = 0.0;
double beta;
double h;
double Nt;
int Ntau;
const int BootstrapMaxIter=1200;
double err;
int Norb;
char *Outfile;

double lmbd0,lmbd1,lmbd_bath,W0;

if (argc<=1){
  Nt=16;
  lmbd0=0.5;
  lmbd1=0.33;
  lmbd_bath=0.025;
  W0=0.125;
  h=0.1;
  Norb=100;
  Outfile="data.dat";
  beta = 13;
  Ntau=150;}
else{
  Nt=atoi(argv[ 1 ]);
  Ntau=atoi(argv[ 2 ]);
  lmbd0=atof(argv[ 3 ]);
  lmbd1=atof(argv[ 4 ]);
  lmbd_bath=atof(argv[ 5 ]);
  W0=atof(argv[ 6 ]);
  h=atof(argv[ 7 ]);
  Norb=atoi(argv[ 8 ]);
  beta=atof(argv[9]);
  Outfile=argv[ 10 ];
  }
        printf( "argc:     %d\n", argc );
        for( int i = 0; i < argc; ++i ) {
         printf( "argv[ %d ] = %s\n", i, argv[ i ] );}

omp_set_num_threads(THREAD_NUM);
// omp_set_num_threads(2);
double WD=3/2*W0;


double tstp;
double Phfreq_w0=0.0001;
double Nphon=1000;
double dw=(WD-Phfreq_w0)/(Nphon-1);

//..................................................
//                input
//..................................................
int SolveOrder;

//..................................................
//                internal
//..................................................
double err_fourier,err_fixpoint;
//.................................................

SolveOrder=5;

double E0=1;
double dxi=2*E0/(Norb-1);

cdmatrix sigg=cdmatrix::Zero(1,1);
sigg(0,0)=1.;

cdmatrix sig3=cdmatrix::Zero(2,2);

sig3(0,0)=1.;
sig3(0,1)=-0.;
sig3(1,0)=0.;
sig3(1,1)=-1.;

cdmatrix sig1=cdmatrix::Zero(2,2);

sig1(0,0)=0.;
sig1(0,1)=1.;
sig1(1,0)=1.;
sig1(1,1)=0.;

cdmatrix sig0=cdmatrix::Zero(2,2);

sig0(0,0)=1.;
sig0(0,1)=0.;
sig0(1,0)=0.;
sig0(1,1)=1.;

cdmatrix sig01=cdmatrix::Zero(2,2);

sig01(0,0)=0.;
sig01(0,1)=1.;
sig01(1,0)=0.;
sig01(1,1)=0.;

cdmatrix sig10=cdmatrix::Zero(2,2);

sig10(0,0)=0.;
sig10(0,1)=0.;
sig10(1,0)=1.;
sig10(1,1)=0.;


CFUNC g_F;
g_F = CFUNC(Nt,2);

double zeta=0.6;
for(int it=-1 ; it<=Nt ; it++) {g_F.set_value(it,sig3);}





CFUNC hbdg_[Norb];

GREEN D(Nt,Ntau,1.,BOSON);
GREEN D_tmp(Nt,Ntau,1.,BOSON);

// #pragma omp parallel for
for(int k=0;k<=Nphon-1;k++)
{
  cntr::green_single_pole_XX(D_tmp,Phfreq_w0+k*dw,beta,h);
  for(int it=-1 ; it<=Nt ; it++) 
  {
    // D_tmp.smul(it,lmbd_bath*dw*abs(cos((Phfreq_w0+k*dw)/W0)-1.)*dxi*2.0/(2*M_PI*0.387965));
    D_tmp.smul(it,2*lmbd_bath*dw*(Phfreq_w0+k*dw)*(Phfreq_w0+k*dw)*dxi*2.0/(WD*WD));
    // #pragma omp critical
    D.incr_timestep(it,D_tmp);
  }
  
}



GREEN Sigma2x2(Nt,Ntau,2,FERMION);
GREEN Sigma2x2_tmp(Nt,Ntau,2,FERMION);
CFUNC delta(Nt,2);

cdmatrix rho_M=cdmatrix::Zero(2,2);


tstp=-1;

GREEN G2x2_[Norb];
GREEN G2x2_tmp[THREAD_NUM];
GREEN Sigma2x2_tmp_[THREAD_NUM];


cdmatrix delta_[THREAD_NUM];
cdmatrix delta__=cdmatrix::Zero(2,2);

for(int k=0;k<=THREAD_NUM-1;k++){
  Sigma2x2_tmp_[k]=GREEN(Nt,Ntau,2,FERMION);
  delta_[k]=cdmatrix::Zero(2,2);
  G2x2_tmp[k]=GREEN(Nt,Ntau,2,FERMION);

}
#pragma omp parallel for
for(int k=0;k<=Norb-1;k++){
  // Sigma2x2_tmp_[k]=GREEN(Nt,Ntau,2,FERMION);
  G2x2_[k]=GREEN(Nt,Ntau,2,FERMION);
  
  cdmatrix H=cdmatrix::Zero(2,2);
  H=(-E0+dxi*k)*sig3+dlt*sig1;
  cntr::green_from_H(G2x2_[k],mu,H,beta,h);
  hbdg_[k] = CFUNC(Nt,2);

}



/////MATSUBARA
chrono::time_point<std::chrono::system_clock> start, end, start_tot, end_tot;
start = std::chrono::system_clock::now();
for(int iter=0;iter<=BootstrapMaxIter;iter++)
    {

    // GREEN_TSTP  Sigma2x2_tmp_[4];
    Sigma2x2.set_timestep_zero(tstp);
    delta__=cdmatrix::Zero(2,2);
    #pragma omp parallel for
    for ( int i =0; i <= Norb-1 ; i ++) { 
        int j=omp_get_thread_num();


        cntr::Bubble2(tstp,Sigma2x2_tmp_[j],0,0,D,0,0,G2x2_[i],0,0); 
        cntr::Bubble2(tstp,Sigma2x2_tmp_[j],1,0,D,0,0,G2x2_[i],1,0); 
        cntr::Bubble2(tstp,Sigma2x2_tmp_[j],0,1,D,0,0,G2x2_[i],0,1); 
        cntr::Bubble2(tstp,Sigma2x2_tmp_[j],1,1,D,0,0,G2x2_[i],1,1); 

        G2x2_[i].density_matrix(tstp,delta_[j]);
        #pragma omp critical
        {
        Sigma2x2.incr_timestep(tstp,Sigma2x2_tmp_[j]);
        delta__+=delta_[j];
        }
    }
      delta.set_value(tstp,delta__);
      Sigma2x2.left_multiply(tstp,g_F);
      Sigma2x2.right_multiply(tstp,g_F);
    
    err=0;
    #pragma omp parallel for
    for ( int i =0; i <= Norb-1 ; i ++) { 
      
      int j=omp_get_thread_num();


      G2x2_tmp[j].set_timestep(tstp,G2x2_[i]);
      hbdg_[i].set_value(tstp,(-E0+dxi*i)*sig3-dxi*lmbd0*(delta__(0,1)*sig01+delta__(1,0)*sig10));
      cntr::dyson_mat(G2x2_[i], 0.0, hbdg_[i], Sigma2x2, beta, SolveOrder, CNTR_MAT_FOURIER);
      
      
      #pragma omp critical
      err += cntr::distance_norm2(tstp,G2x2_[i],G2x2_tmp[j]);     
    }


    if(err<1e-6 && iter>2) break;
    Sigma2x2.density_matrix(tstp,rho_M);
    
     cout<<"Matsubara niter:"<<iter<<"   error:"<<err<<"  gap="<<dxi*lmbd0*delta__(0,1)<<"   "<<delta__(0,0)<<endl;

    }

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  cout << "Time  = " << elapsed_seconds.count() << "s\n\n";





GREEN Sigma2x2_tmp1(Nt,Ntau,2,FERMION);


cntr::set_tk_from_mat(Sigma2x2,SolveOrder);
#pragma omp parallel for
for(int k=0;k<=Norb-1;k++)
  {
  cntr::set_tk_from_mat(G2x2_[k],SolveOrder);
  }

for (int iter = 0; iter <= BootstrapMaxIter; iter++) {
    for(int it=-1 ; it<=SolveOrder ; it++) {
      //  Gtemp.set_timestep(it,G);   

        Sigma2x2_tmp1.set_timestep_zero(it);
        Sigma2x2_tmp.set_timestep_zero(it);
       }
 





  for(int tstp=0 ; tstp<=SolveOrder ; tstp++) {
      Sigma2x2_tmp1.set_timestep_zero(tstp);
      // Sigma2x2_tmp.set_timestep_zero(tstp);
      delta__=cdmatrix::Zero(2,2);
    #pragma omp parallel for
    for ( int i =0; i <= Norb-1 ; i ++) { 
      int j=omp_get_thread_num();


        cntr::Bubble2(tstp,Sigma2x2_tmp_[j],0,0,D,0,0,G2x2_[i],0,0); 
        cntr::Bubble2(tstp,Sigma2x2_tmp_[j],1,0,D,0,0,G2x2_[i],1,0); 
        cntr::Bubble2(tstp,Sigma2x2_tmp_[j],0,1,D,0,0,G2x2_[i],0,1); 
        cntr::Bubble2(tstp,Sigma2x2_tmp_[j],1,1,D,0,0,G2x2_[i],1,1); 

      G2x2_[i].density_matrix(tstp,delta_[j]);
      #pragma omp critical
      {
      Sigma2x2_tmp1.incr_timestep(tstp,Sigma2x2_tmp_[j]);
      delta__+=delta_[j];
      }
    }
      Sigma2x2_tmp1.left_multiply(tstp,g_F);
      Sigma2x2_tmp1.right_multiply(tstp,g_F);
      Sigma2x2.set_timestep(tstp,Sigma2x2_tmp1);

      Sigma2x2.density_matrix(tstp,rho_M);
      delta.set_value(tstp,delta__);
      for ( int i =0; i <= Norb-1 ; i ++) { hbdg_[i].set_value(tstp,(-E0+dxi*i)*sig3-dxi*lmbd1*(delta__(0,1)*sig01+delta__(1,0)*sig10));}

  }
    err=0;
    #pragma omp parallel for
    for ( int i =0; i <= Norb-1 ; i ++) { 
      int j=omp_get_thread_num();

        G2x2_tmp[j].set_timestep(tstp,G2x2_[i]);
      
      

      // hbdg_[i].set_value(tstp,(-E0+dxi*i)*sig3-0.5*dxi*lmbd1*(delta__(0,1)+delta__(1,0))*sig1);
      cntr::dyson_start(G2x2_[i], 0.0, hbdg_[i], Sigma2x2, beta,h, SolveOrder);
      err += cntr::distance_norm2(tstp,G2x2_[i],G2x2_tmp[j]);
   
    }
    
    cout << "bootstrap iteration : " << iter << "  |  Error = " << err <<"  gap="<<dxi*lmbd1*delta__(0,1)<< endl;
    if(err<1e-10 && iter>2){
    // bootstrap_converged=true;
    break;
    }

    
    
    
    }

    int CorrectorSteps=100;
    start = std::chrono::system_clock::now();



  for(tstp = SolveOrder+1; tstp <= Nt; tstp++){

    cntr::extrapolate_timestep(tstp-1,Sigma2x2,SolveOrder);
    #pragma omp parallel for
    for ( int i =0; i <= Norb-1 ; i ++) {cntr::extrapolate_timestep(tstp-1,G2x2_[i],SolveOrder);}
      

      for (int iter=0; iter < CorrectorSteps; iter++){


      Sigma2x2_tmp1.set_timestep_zero(tstp);
      delta__=cdmatrix::Zero(2,2);

      #pragma omp parallel for
       for ( int i =0; i <= Norb-1 ; i ++) { 

        int j=omp_get_thread_num();



        cntr::Bubble2(tstp,Sigma2x2_tmp_[j],0,0,D,0,0,G2x2_[i],0,0); 
        cntr::Bubble2(tstp,Sigma2x2_tmp_[j],1,0,D,0,0,G2x2_[i],1,0); 
        cntr::Bubble2(tstp,Sigma2x2_tmp_[j],0,1,D,0,0,G2x2_[i],0,1); 
        cntr::Bubble2(tstp,Sigma2x2_tmp_[j],1,1,D,0,0,G2x2_[i],1,1); 
        G2x2_[i].density_matrix(tstp,delta_[j]);


      #pragma omp critical
      {
       Sigma2x2_tmp1.incr_timestep(tstp,Sigma2x2_tmp_[j]);
       delta__+=delta_[j];
      }
    }
      Sigma2x2_tmp1.left_multiply(tstp,g_F);
      Sigma2x2_tmp1.right_multiply(tstp,g_F);
      Sigma2x2.set_timestep(tstp,Sigma2x2_tmp1);
      // Sigma2x2.smul(tstp,1);
      Sigma2x2.density_matrix(tstp,rho_M);
      delta.set_value(tstp,delta__);

      err=0;
       #pragma omp parallel for
       for ( int i =0; i <= Norb-1 ; i ++) { 
        int j=omp_get_thread_num();

        G2x2_tmp[j].set_timestep(tstp,G2x2_[i]);



      hbdg_[i].set_value(tstp,(-E0+dxi*i)*sig3-dxi*lmbd1*(delta__(0,1)*sig01+delta__(1,0)*sig10));

      cntr::dyson_timestep(tstp, G2x2_[i], 0.0, hbdg_[i], Sigma2x2, beta,h, SolveOrder);
      err += cntr::distance_norm2(tstp,G2x2_[i],G2x2_tmp[j]);
       
    
    }
    
    
    cout << "timestep tstp="<<tstp<<"iter=" << iter << "  |  Error = " << err <<"  gap="<<dxi*lmbd1*delta__(0,1)<< endl;
    if(err<1e-6 && iter>2) break;
    
    }
    }

        

    








// Sigma2x2.print_to_file("Sigma1.dat",16);
// G2x2_[20].print_to_file("G.dat",16);
// Sigma2x2.print_to_file(Outfile,16);

delta.smul(-dxi*lmbd1);

CFUNC delta01;
delta01 = CFUNC(Nt,1);
delta.get_matrixelement(0,1,delta01);
delta01.print_to_file(Outfile,16);


delta01 = CFUNC(Nt,2);
const char *cstr;
string outfl=Outfile;
// #pragma omp parallel for
for(int k=0;k<=Norb-1;k++){
  outfl=Outfile;
  // outfl.append()
  if(k<10){ cstr =(outfl.append("_00"+to_string(k))).c_str();}
  else if(k<100){ cstr =(outfl.append("_0"+to_string(k))).c_str();}
  else {cstr =(outfl.append('_'+to_string(k))).c_str();};
  


  // cstr =(to_string(beta)+to_string(k)).c_str();
  // cout<<cstr<<endl;

  for(int it=-1 ; it<=Nt ; it++) {
    G2x2_[k].density_matrix(it,rho_M);

    delta01.set_value(it,rho_M);
    }

  delta01.print_to_file(cstr,16);


}



// D.print_to_file("D.dat",16);



// CFUNC rho_M_F;
// rho_M_F = CFUNC(Nt,2);
// outfl=Outfile;
// cstr =outfl.append("S").c_str();
// for(int it=-1 ; it<=Nt ; it++) {
//     Sigma2x2.get_ret(it,it,rho_M);
//     rho_M_F.set_value(it,rho_M);
//     }
// rho_M_F.print_to_file(cstr,16);


outfl=Outfile;
cstr =outfl.append("S").c_str();

Sigma2x2.print_to_file(cstr,16);



return 0;
}
