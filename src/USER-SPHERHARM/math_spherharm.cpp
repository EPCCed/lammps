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

#include "math_spherharm.h"
#include "math_const.h"
#include "gaussquad_const.h"

using namespace LAMMPS_NS::GaussquadConst;

namespace MathSpherharm {

/* ----------------------------------------------------------------------
  Calculate the Associated Legendre polynomials (generic)
------------------------------------------------------------------------- */

  double plegendre(const int l, const int m, const double x) {

    int i,ll;
    double fact,oldfact,pll,pmm,pmmp1,omx2;
//    if (m < 0 || m > l || std::abs(x) > 1.0)
//      throw std::invalid_argument( "Bad arguments in routine plgndr" );
    pmm=1.0;
    if (m > 0) {
      omx2=(1.0-x)*(1.0+x);
      fact=1.0;
      for (i=1;i<=m;i++) {
        pmm *= omx2*fact/(fact+1.0);
        fact += 2.0;
      }
    }
    pmm=sqrt((2.0*m+1.0)*pmm/(4.0*LAMMPS_NS::MathConst::MY_PI));
    if (m & 1)
      pmm=-pmm;
    if (l == m)
      return pmm;
    else {
      pmmp1=x*sqrt(2.0*m+3.0)*pmm;
      if (l == (m+1))
        return pmmp1;
      else {
        oldfact=sqrt(2.0*m+3.0);
        for (ll=m+2;ll<=l;ll++) {
          fact=sqrt((4.0*ll*ll-1.0)/(ll*ll-m*m));
          pll=(x*pmmp1-pmm/oldfact)*fact;
          oldfact=fact;
          pmm=pmmp1;
          pmmp1=pll;
        }
        return pll;
      }
    }
  }

/* ----------------------------------------------------------------------
  Calculating the Associated Legendre polynomials (when n=m)
------------------------------------------------------------------------- */
  double plegendre_nn(const int l, const double x, const double Pnm_nn) {

    double ll, llm1, fact;

    ll = (double) l;
    llm1 = 2.0*(ll-1.0);
//    if (std::abs(x) > 1.0)
//      throw std::invalid_argument( "Bad arguments in routine plgndr" );
    fact = sqrt((llm1 + 3.0)/(llm1 + 2.0));
    return -sqrt(1.0-(x*x)) * fact * Pnm_nn;
  }

/* ----------------------------------------------------------------------
  Calculating the Associated Legendre polynomials (recursion)
------------------------------------------------------------------------- */
  double plegendre_recycle(const int l, const int m, const double x, const double pnm_m1, const double pnm_m2) {

    double fact,oldfact, ll, mm, pmn;

    ll = (double) l;
    mm = (double) m;
    fact = sqrt((4.0*ll*ll-1.0)/(ll*ll-mm*mm));
    oldfact = sqrt((4.0*(ll-1.0)*(ll-1.0)-1.0)/((ll-1.0)*(ll-1.0)-mm*mm));
    pmn=(x*pnm_m1-pnm_m2/oldfact)*fact;
    return pmn;
  }

  double plgndr(const int l, const int m, const double x)
  {
    if (m < 0 || m > l || std::abs(x) > 1.0)
      return 0;
    double prod=1.0;
    for (int j=l-m+1;j<=l+m;j++)
      prod *= j;
    return sqrt(4.0*LAMMPS_NS::MathConst::MY_PI*prod/(2*l+1))*plegendre(l,m,x);
  }

/* ----------------------------------------------------------------------
  Following methods are used for calculating the nodes and weights of
  Gaussian Quadrature
------------------------------------------------------------------------- */
// This function computes the kth zero of the BesselJ(0,x)
  double besseljzero(int k)
  {
    if(k > 20)
    {
      double z = LAMMPS_NS::MathConst::MY_PI*(k-0.25);
      double r = 1.0/z;
      double r2 = r*r;
      z = z + r*(0.125+r2*(-0.807291666666666666666666666667e-1+r2*(0.246028645833333333333333333333+r2*(-1.82443876720610119047619047619+r2*(25.3364147973439050099206349206+r2*(-567.644412135183381139802038240+r2*(18690.4765282320653831636345064+r2*(-8.49353580299148769921876983660e5+5.09225462402226769498681286758e7*r2))))))));
      return z;
    }
    else
    {
      return JZ[k-1];
    }
  }


// This function computes the square of BesselJ(1, BesselZero(0,k))
  double besselj1squared(int k)
  {
    if(k > 21)
    {
      double x = 1.0/(k-0.25);
      double x2 = x*x;
      return x * (0.202642367284675542887758926420 + x2*x2*(-0.303380429711290253026202643516e-3 + x2*(0.198924364245969295201137972743e-3 + x2*(-0.228969902772111653038747229723e-3+x2*(0.433710719130746277915572905025e-3+x2*(-0.123632349727175414724737657367e-2+x2*(0.496101423268883102872271417616e-2+x2*(-0.266837393702323757700998557826e-1+.185395398206345628711318848386*x2))))))));
    }
    else
    {
      return J1[k-1];
    }
  }


// Compute a node-weight pair, with k limited to half the range
  QuadPair GLPairS(size_t n, size_t k)
  {
    // First get the Bessel zero
    double w = 1.0/(n+0.5);
    double nu = besseljzero(k);
    double theta = w*nu;
    double x = theta*theta;

    // Get the asymptotic BesselJ(1,nu) squared
    double B = besselj1squared(k);

    // Get the Chebyshev interpolants for the nodes...
    double SF1T = (((((-1.29052996274280508473467968379e-12*x +2.40724685864330121825976175184e-10)*x -3.13148654635992041468855740012e-8)*x +0.275573168962061235623801563453e-5)*x -0.148809523713909147898955880165e-3)*x +0.416666666665193394525296923981e-2)*x -0.416666666666662959639712457549e-1;
    double SF2T = (((((+2.20639421781871003734786884322e-9*x  -7.53036771373769326811030753538e-8)*x  +0.161969259453836261731700382098e-5)*x -0.253300326008232025914059965302e-4)*x +0.282116886057560434805998583817e-3)*x -0.209022248387852902722635654229e-2)*x +0.815972221772932265640401128517e-2;
    double SF3T = (((((-2.97058225375526229899781956673e-8*x  +5.55845330223796209655886325712e-7)*x  -0.567797841356833081642185432056e-5)*x +0.418498100329504574443885193835e-4)*x -0.251395293283965914823026348764e-3)*x +0.128654198542845137196151147483e-2)*x -0.416012165620204364833694266818e-2;

    // ...and for the weights
    double WSF1T = ((((((((-2.20902861044616638398573427475e-14*x+2.30365726860377376873232578871e-12)*x-1.75257700735423807659851042318e-10)*x+1.03756066927916795821098009353e-8)*x-4.63968647553221331251529631098e-7)*x+0.149644593625028648361395938176e-4)*x-0.326278659594412170300449074873e-3)*x+0.436507936507598105249726413120e-2)*x-0.305555555555553028279487898503e-1)*x+0.833333333333333302184063103900e-1;
    double WSF2T = (((((((+3.63117412152654783455929483029e-12*x+7.67643545069893130779501844323e-11)*x-7.12912857233642220650643150625e-9)*x+2.11483880685947151466370130277e-7)*x-0.381817918680045468483009307090e-5)*x+0.465969530694968391417927388162e-4)*x-0.407297185611335764191683161117e-3)*x+0.268959435694729660779984493795e-2)*x-0.111111111111214923138249347172e-1;
    double WSF3T = (((((((+2.01826791256703301806643264922e-9*x-4.38647122520206649251063212545e-8)*x+5.08898347288671653137451093208e-7)*x-0.397933316519135275712977531366e-5)*x+0.200559326396458326778521795392e-4)*x-0.422888059282921161626339411388e-4)*x-0.105646050254076140548678457002e-3)*x-0.947969308958577323145923317955e-4)*x+0.656966489926484797412985260842e-2;

    // Then refine with the paper expansions
    double NuoSin = nu/sin(theta);
    double BNuoSin = B*NuoSin;
    double WInvSinc = w*w*NuoSin;
    double WIS2 = WInvSinc*WInvSinc;

    // Finally compute the node and the weight
    theta = w*(nu + theta * WInvSinc * (SF1T + WIS2*(SF2T + WIS2*SF3T)));
    double Deno = BNuoSin + BNuoSin * WIS2*(WSF1T + WIS2*(WSF2T + WIS2*WSF3T));
    double weight = (2.0*w)/Deno;
    return QuadPair(theta,weight);
  }


// Returns tabulated theta and weight values: valid for l <= 100
  QuadPair GLPairTabulated(size_t l, size_t k)
  {
    // Odd Legendre degree
    if(l & 1)
    {
      size_t l2 = (l-1)/2;
      if(k == l2)
        return(QuadPair(LAMMPS_NS::MathConst::MY_PI/2, 2.0/(Cl[l]*Cl[l])));
      else if(k < l2)
        return(QuadPair(OddThetaZeros[l2-1][l2-k-1],OddWeights[l2-1][l2-k-1]));
      else
        return(QuadPair(LAMMPS_NS::MathConst::MY_PI-OddThetaZeros[l2-1][k-l2-1],OddWeights[l2-1][k-l2-1]));
    }
      // Even Legendre degree
    else
    {
      size_t l2 = l/2;
      if(k < l2)
        return(QuadPair(EvenThetaZeros[l2-1][l2-k-1],EvenWeights[l2-1][l2-k-1]));
      else
        return(QuadPair(LAMMPS_NS::MathConst::MY_PI-EvenThetaZeros[l2-1][k-l2],EvenWeights[l2-1][k-l2]));
    }
  }


// This function computes the kth GL pair of an n-point rule
  QuadPair GLPair(size_t n, size_t k)
  {
    // Sanity check [also implies l > 0]
    if(n < 101)
      return(GLPairTabulated(n, k-1));
    else
    {
      if((2*k-1) > n)
      {
        QuadPair P = GLPairS(n, n-k+1);
        P.theta = LAMMPS_NS::MathConst::MY_PI - P.theta;
        return P;
      }
      else return GLPairS(n, k);
    }
  }

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   factorial n, wrapper for precomputed table
------------------------------------------------------------------------- */

  double factorial(int n)
  {
    if (n < 0 || n > nmaxfactorial)
      return -1.0;

    return nfac_table[n];
  }
}
