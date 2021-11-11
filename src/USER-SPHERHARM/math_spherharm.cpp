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
 * factorial n, wrapper for precomputed table
------------------------------------------------------------------------- */

  double factorial(int n)
  {
    if (n < 0 || n > nmaxfactorial)
      return -1.0;

    return nfac_table[n];
  }

  /* ----------------------------------------------------------------------
   * Return 0 for 2 or 1 intersection. Return 1 for no intersections. Points
   * are measured as distances along the line from its origin.
  ------------------------------------------------------------------------- */
  int line_sphere_intersection(const double rad, const double circcentre[3], const double linenorm[3],
                               const double lineorigin[3], double &sol1, double &sol2){

    double omc[3], discr;
    MathExtra::sub3(lineorigin, circcentre, omc);
    double udomc = MathExtra::dot3(linenorm, omc);
    discr = (udomc*udomc) -  MathExtra::lensq3(omc) + (rad*rad);

    if (discr > 0.0){
      double sqd = std::sqrt(discr);
      sol1 = -udomc + sqd;
      sol2 = -udomc - sqd;
      return 0;
    }
    else if (discr == 0.0){
      double sqd = std::sqrt(discr);
      sol1 = -udomc + sqd;
      sol2 = sol1;
      return 0;
    }
    else{
      return 1;
    }
  }

    /* ----------------------------------------------------------------------
   * Return 0 for intersection. Return 1 for no intersections. Points
   * are measured as distances along the line from its origin.
  ------------------------------------------------------------------------- */
  int line_plane_intersection(double (&p0)[3], double (&l0)[3], double (&l)[3], double (&n)[3], double &sol){

    double numer, denom;
    denom = MathExtra::dot3(l, n);
    if (denom==0) return 1;
    numer = (p0[0]-l0[0])*n[0] + (p0[1]-l0[1])*n[1] + (p0[2]-l0[2])*n[2];
    sol = numer/denom;
    return 0;
  }

  /* ----------------------------------------------------------------------
 * Return 0,1 corresponding success and failure. Points
 * are measured as distances along the line from its origin. Cylinder
   * must be located along z-axis at (0,0)
------------------------------------------------------------------------- */
  int line_cylinder_intersection(const double xi[3], const double (&unit_line_normal)[3], double &t1,
          double &t2, double cylradius){

    double aa, bb, cc, discrim;

    // Parameterize line from particle centre to surface point and plug into the equation for a cylinder to find
    // intersections points (solve quadratic equation)
    aa = unit_line_normal[0]*unit_line_normal[0] + unit_line_normal[1]*unit_line_normal[1]; // aa, bb, cc for quadratic equation
    bb = 2.0*(xi[0]*unit_line_normal[0] + xi[1]*unit_line_normal[1]);
    cc = xi[0]*xi[0] + xi[1]*xi[1] - cylradius*cylradius;
    discrim = bb*bb - 4.0*aa*cc;

    if (discrim > 0.0){
      discrim = std::sqrt(discrim);
      t1 = (-bb + discrim) / (2.0*aa); // first solution
      t2 = (-bb - discrim) / (2.0*aa); // first solution
      return 0;
    }
    else if (discrim == 0.0){ //
      discrim = std::sqrt(discrim);
      t1 = (-bb + discrim) / (2.0*aa); // first solution
      t2 = t1;
      return 0;
    }
    else{
      return 1;
    }
  }

  // contact point cp is relative to particle a's centre
  int get_contact_point_plane(double rada, double xi[3], double (&linenorm)[3], double (&lineorigin)[3], double
  (&p0)[3], double (&cp)[3]){
    int not_ok;
    double sol1, sol2;
    double n[3];

    MathExtra::copy3(p0, n); // Planes are always perpendicular to the particle centre, can normalise for the plane normal
    MathExtra::normalize3(n, n); // Ensuring that the plane normal is a unit vector

    //Intersection of the contact line with particle A's bounding sphere
    not_ok = line_sphere_intersection(rada, xi, linenorm, lineorigin, sol1, sol2);
    if (not_ok) return not_ok;

    cp[0] = lineorigin[0] + linenorm[0]*sol1; // finding the point's 3d coordinates
    cp[1] = lineorigin[1] + linenorm[1]*sol1;
    cp[2] = lineorigin[2] + linenorm[2]*sol1;

    // Save the point that's on the positive side of the wall plane (i.e. outside the boundary)(assumes boundary faces
    // outwards
    if ((n[0]*cp[0] + n[1]*cp[1] + n[2]*cp[2] - n[0]*p0[0] + n[1]*p0[1] + n[2]*p0[2])<0) sol1 = sol2;

    //Intersection of the contact line with plane
    not_ok = line_plane_intersection(p0, lineorigin, linenorm, n, sol2);
    if (not_ok) return not_ok;

    sol1 = 0.5*(sol1+sol2); // Average of the bounds
    cp[0] = lineorigin[0] + linenorm[0]*sol1; // finding the contact point's 3d coordinates
    cp[1] = lineorigin[1] + linenorm[1]*sol1;
    cp[2] = lineorigin[2] + linenorm[2]*sol1;
    return 0;
  }

  // contact point cp is relative to particle a's centre
  int get_contact_point_cylinder(double rada, double xi[3], double (&linenorm)[3], double(&lineorigin)[3], double
  (&cp)[3], double cylrad, bool inside){

    int not_ok;
    double sol1, sol2;
    double sol3, sol4;
    double tmp;
    double lineorigin_global[3];

    MathExtra::add3(lineorigin, xi, lineorigin_global);

    //Intersection of the contact line with particle A's bounding sphere
    not_ok = line_sphere_intersection(rada, xi, linenorm, lineorigin_global, sol1, sol2);
    if (not_ok) return not_ok;

    // Cylinder must be at (x=y=0) so the line origin must be in global coordinates
    not_ok = line_cylinder_intersection(lineorigin_global, linenorm, sol3, sol4, cylrad);
    if (not_ok) return not_ok;

    std::cout <<"sol1 " << sol1 << std::endl;
    std::cout <<"sol2 " << sol2 << std::endl;
    std::cout <<"sol3 " << sol3 << std::endl;
    std::cout <<"sol4 " << sol4 << std::endl;

    // Now need to find the inner values of the 4 solutions.
    if (sol1 > sol2) { tmp = sol1; sol1 = sol2; sol2 = tmp; }
    if (sol3 > sol4) { tmp = sol3; sol3 = sol4; sol4 = tmp; }
    if (sol1 > sol3) { tmp = sol1; sol1 = sol3; sol3 = tmp; }
    if (sol2 > sol4) { tmp = sol2; sol2 = sol4; sol4 = tmp; }
    if (sol2 > sol3) { tmp = sol2; sol2 = sol3; sol3 = tmp; } // middle of sorted values is sol2 and sol3

    if (inside){
      sol1 = 0.5*(sol1+sol2); // Average of the minimum bounds
    } else sol1 = 0.5*(sol2+sol3); // Average of the bounds
    cp[0] = lineorigin[0] + linenorm[0]*sol1; // finding the contact point's 3d coordinates
    cp[1] = lineorigin[1] + linenorm[1]*sol1;
    cp[2] = lineorigin[2] + linenorm[2]*sol1;

    return 0;
  }


  // Get the quaternion between the north pole and a given vector
  void get_contact_quat(double (&xvecdist)[3], double (&quat)[4]) {
    double vert_unit_vec[3], cross_vec[3], c;
    // North pole unit vector, points generated are with reference to this point
    vert_unit_vec[0] = 0.0;
    vert_unit_vec[1] = 0.0;
    vert_unit_vec[2] = 1.0;
    if (xvecdist[0]==0.0 and xvecdist[1]==0.0){
      if(xvecdist[2]<0.0) { //rotation to south pole
        quat[1] = 1.0;
        quat[2] = 0.0;
        quat[3] = 0.0;
        quat[0] = 0.0;
      }
      else{
        quat[1] = 0.0; //identity quaternion, no rotation, default case
        quat[2] = 0.0;
        quat[3] = 0.0;
        quat[0] = 1.0;
      }
    }
    else {
      c = MathExtra::dot3(vert_unit_vec, xvecdist);
      MathExtra::cross3(vert_unit_vec, xvecdist, cross_vec);
      quat[1] = cross_vec[0];
      quat[2] = cross_vec[1];
      quat[3] = cross_vec[2];
      quat[0] = sqrt(MathExtra::lensq3(vert_unit_vec) * MathExtra::lensq3(xvecdist)) + c;
      MathExtra::qnormalize(quat);
    }
  }

  // Get the volume of overlap between two spheres
  double get_sphere_overlap_volume(double r1, double r2, double d) {
    return LAMMPS_NS::MathConst::MY_PI*(r1+r2-d)*(r1+r2-d)*
      (d*d + 2.0*d*(r1+r2) - 3.0*(r1-r2)*(r1-r2))/(12.0*d);
  }

  // Calculate the interseciton between a line and an ellipsoid. It's assumed that the line generates from the centre
  // of another ellipsoid (see the selection between t1 and t2). Algorithm from:
  // http://www.illusioncatalyst.com/notes_files/mathematics/line_nu_sphere_intersection.php
  int line_ellipsoid_intersection(const double elipsoid_centre[3], const double elipse_x_axis[3],
                                  const double elipse_y_axis[3],const double elipse_z_axis[3],
                                  const double line_centre[3], const double line_normal[3],
                                  double &t){

    double M[4][4], Mi[4][4];
    double C[4], L0[4];
    double Cp[4], L0p[4];
    double vp[3], w[3];
    double T[4][4] = {0.};
    double R[4][4] = {0.};
    double S[4][4] = {0.};
    double a,b,c,discrim,t1,t2;
    double lenvec;

    // Converting to shape (4,) so can be multiplied by the transformation matrix
    C[0] = elipsoid_centre[0];
    C[1] = elipsoid_centre[1];
    C[2] = elipsoid_centre[2];
    C[3] = 1.0;

    // Converting to shape (4,) so can be multiplied by the transformation matrix
    L0[0] = line_centre[0];
    L0[1] = line_centre[1];
    L0[2] = line_centre[2];
    L0[3] = 1.0;

    // Creating the transformation matrices
    //
    // Translation
    T[0][0] = T[1][1] = T[2][2] = T[3][3] = 1.0;
    T[0][3] = C[0];
    T[1][3] = C[1];
    T[2][3] = C[2];

    // Rotation
    lenvec = MathExtra::len3(elipse_x_axis);
    R[0][0] = elipse_x_axis[0]/lenvec;
    R[1][0] = elipse_x_axis[1]/lenvec;
    R[2][0] = elipse_x_axis[2]/lenvec;
    lenvec = MathExtra::len3(elipse_y_axis);
    R[0][1] = elipse_y_axis[0]/lenvec;
    R[1][1] = elipse_y_axis[1]/lenvec;
    R[2][1] = elipse_y_axis[2]/lenvec;
    lenvec = MathExtra::len3(elipse_z_axis);
    R[0][2] = elipse_z_axis[0]/lenvec;
    R[1][2] = elipse_z_axis[1]/lenvec;
    R[2][2] = elipse_z_axis[2]/lenvec;
    R[3][3] = 1.0;

    // Scaling
    S[0][0] = MathExtra::len3(elipse_x_axis);
    S[1][1] = MathExtra::len3(elipse_y_axis);
    S[2][2] = MathExtra::len3(elipse_z_axis);
    S[3][3] = 1.0;

    // Combining for M = TRS, Mi = inverse(M), recycling Mi
    times4(T,R,Mi);
    times4(Mi,S,M);
    invert4(M,Mi);

    std::cout<< std::endl;
    std::cout<< Mi[0][0] << " " << Mi[0][1] << " " << Mi[0][2] << " " << Mi[0][3] << " " <<std::endl;
    std::cout<< Mi[1][0] << " " << Mi[1][1] << " " << Mi[1][2] << " " << Mi[1][3] << " " << std::endl;
    std::cout<< Mi[2][0] << " " << Mi[2][1] << " " << Mi[2][2] << " " << Mi[2][3] << " " << std::endl;
    std::cout<< Mi[3][0] << " " << Mi[3][1] << " " << Mi[3][2] << " " << Mi[3][3] << " " << std::endl;

    // Transforming ellipsoid centre and point on line
    matvec4(Mi,C,Cp);
    matvec4(Mi,L0,L0p);

    // Converting to shape (4,) so can be multiplied by the transformation matrix, recycling L0
    L0[0] = line_normal[0];
    L0[1] = line_normal[1];
    L0[2] = line_normal[2];
    L0[3] = 1.0;

    // Transforming the line normal and then converting back to shape (3,), recycling C
    matvec4(Mi,L0,C);
    vp[0] = C[0];
    vp[1] = C[1];
    vp[2] = C[2];

    w[0] = L0p[0] - Cp[0];
    w[1] = L0p[1] - Cp[1];
    w[2] = L0p[2] - Cp[2];

    std::cout<< std::endl;
    std::cout<< elipsoid_centre[0] << " " << elipsoid_centre[1] << " " << elipsoid_centre[2] << " " << std::endl;
    std::cout<< Cp[0] << " " << Cp[1] << " " << Cp[2] << " " << std::endl;
    std::cout<< L0p[0] << " " << L0p[1] << " " << L0p[2] << " " << std::endl;
    std::cout<< vp[0] << " " << vp[1] << " " << vp[2] << " " << std::endl;
    std::cout<< w[0] << " " << w[1] << " " << w[2] << " " << std::endl;

    // Quadratic terms in t
    a = MathExtra::dot3(vp,vp);
    b = 2.0*MathExtra::dot3(vp,w);
    c = MathExtra::dot3(w,w) - 1.0;
    discrim = b*b - 4.0*a*c;

    if (discrim>0.0){
      t1 = (-b + std::sqrt(discrim))/(2.0 * a);
      t2 = (-b - std::sqrt(discrim))/(2.0 * a);
      // by virtue of the problem, one of these must be negative, i.e. away from the centre is the opposite direction
      // of the contact, we want the positive value, i.e in the direction of the intersection.
      t = t1 < t2 ? t1 : t2;
      std::cout << std::endl << std::endl << std::endl << "T's " << t1 << " " << t2<< std::endl<< std::endl <<
      std::endl;
      return 0;
    }
    else if(discrim==0.0){
      t = -b / (2.0*a);
      return 0;
    }
    else return 1;

  }

} //end namespace