+----------------------------------------+------------------------------------+------------------------------------------+
| :doc:`General commands <Commands_all>` | :doc:`Fix styles <Commands_fix>`   | :doc:`Compute styles <Commands_compute>` |
+----------------------------------------+------------------------------------+------------------------------------------+
| :doc:`Pair styles <Commands_pair>`     | :doc:`Bond styles <Commands_bond>` | :ref:`Angle styles <angle>`              |
+----------------------------------------+------------------------------------+------------------------------------------+
| :ref:`Dihedral styles <dihedral>`      | :ref:`Improper styles <improper>`  | :doc:`KSpace styles <Commands_kspace>`   |
+----------------------------------------+------------------------------------+------------------------------------------+

Pair\_style potentials
======================

All LAMMPS :doc:`pair\_style <pair_style>` commands.  Some styles have
accelerated versions.  This is indicated by additional letters in
parenthesis: g = GPU, i = USER-INTEL, k = KOKKOS, o = USER-OMP, t =
OPT.

+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`none <pair_none>`                                      | :doc:`zero <pair_zero>`                                           | :doc:`hybrid (k) <pair_hybrid>`                     | :doc:`hybrid/overlay (k) <pair_hybrid>`                     |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
|                                                              |                                                                   |                                                     |                                                             |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`adp (o) <pair_adp>`                                    | :doc:`agni (o) <pair_agni>`                                       | :doc:`airebo (io) <pair_airebo>`                    | :doc:`airebo/morse (io) <pair_airebo>`                      |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`atm <pair_atm>`                                        | :doc:`awpmd/cut <pair_awpmd>`                                     | :doc:`beck (go) <pair_beck>`                        | :doc:`body/nparticle <pair_body_nparticle>`                 |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`body/rounded/polygon <pair_body_rounded_polygon>`      | :doc:`body/rounded/polyhedron <pair_body_rounded_polyhedron>`     | :doc:`bop <pair_bop>`                               | :doc:`born (go) <pair_born>`                                |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`born/coul/dsf <pair_born>`                             | :doc:`born/coul/dsf/cs <pair_cs>`                                 | :doc:`born/coul/long (go) <pair_born>`              | :doc:`born/coul/long/cs (g) <pair_cs>`                      |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`born/coul/msm (o) <pair_born>`                         | :doc:`born/coul/wolf (go) <pair_born>`                            | :doc:`born/coul/wolf/cs (g) <pair_cs>`              | :doc:`brownian (o) <pair_brownian>`                         |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`brownian/poly (o) <pair_brownian>`                     | :doc:`buck (giko) <pair_buck>`                                    | :doc:`buck/coul/cut (giko) <pair_buck>`             | :doc:`buck/coul/long (giko) <pair_buck>`                    |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`buck/coul/long/cs <pair_cs>`                           | :doc:`buck/coul/msm (o) <pair_buck>`                              | :doc:`buck/long/coul/long (o) <pair_buck_long>`     | :doc:`buck/mdf <pair_mdf>`                                  |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`buck6d/coul/gauss/dsf <pair_buck6d_coul_gauss>`        | :doc:`buck6d/coul/gauss/long <pair_buck6d_coul_gauss>`            | :doc:`colloid (go) <pair_colloid>`                  | :doc:`comb (o) <pair_comb>`                                 |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`comb3 <pair_comb>`                                     | :doc:`cosine/squared <pair_cosine_squared>`                       | :doc:`coul/cut (gko) <pair_coul>`                   | :doc:`coul/cut/soft (o) <pair_fep_soft>`                    |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`coul/debye (gko) <pair_coul>`                          | :doc:`coul/diel (o) <pair_coul_diel>`                             | :doc:`coul/dsf (gko) <pair_coul>`                   | :doc:`coul/long (gko) <pair_coul>`                          |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`coul/long/cs (g) <pair_cs>`                            | :doc:`coul/long/soft (o) <pair_fep_soft>`                         | :doc:`coul/msm (o) <pair_coul>`                     | :doc:`coul/shield <pair_coul_shield>`                       |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`coul/streitz <pair_coul>`                              | :doc:`coul/wolf (ko) <pair_coul>`                                 | :doc:`coul/wolf/cs <pair_cs>`                       | :doc:`dpd (gio) <pair_dpd>`                                 |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`dpd/fdt <pair_dpd_fdt>`                                | :doc:`dpd/fdt/energy (k) <pair_dpd_fdt>`                          | :doc:`dpd/tstat (go) <pair_dpd>`                    | :doc:`dsmc <pair_dsmc>`                                     |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`e3b <pair_e3b>`                                        | :doc:`drip <pair_drip>`                                           | :doc:`eam (gikot) <pair_eam>`                       | :doc:`eam/alloy (gikot) <pair_eam>`                         |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`eam/cd (o) <pair_eam>`                                 | :doc:`eam/cd/old (o) <pair_eam>`                                  | :doc:`eam/fs (gikot) <pair_eam>`                    | :doc:`edip (o) <pair_edip>`                                 |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`edip/multi <pair_edip>`                                | :doc:`edpd <pair_meso>`                                           | :doc:`eff/cut <pair_eff>`                           | :doc:`eim (o) <pair_eim>`                                   |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`exp6/rx (k) <pair_exp6_rx>`                            | :doc:`extep <pair_extep>`                                         | :doc:`gauss (go) <pair_gauss>`                      | :doc:`gauss/cut (o) <pair_gauss>`                           |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`gayberne (gio) <pair_gayberne>`                        | :doc:`gran/hertz/history (o) <pair_gran>`                         | :doc:`gran/hooke (o) <pair_gran>`                   | :doc:`gran/hooke/history (ko) <pair_gran>`                  |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`granular <pair_granular>`                              | :doc:`gw <pair_gw>`                                               | :doc:`gw/zbl <pair_gw>`                             | :doc:`hbond/dreiding/lj (o) <pair_hbond_dreiding>`          |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`hbond/dreiding/morse (o) <pair_hbond_dreiding>`        | :doc:`ilp/graphene/hbn <pair_ilp_graphene_hbn>`                   | :doc:`kim <pair_kim>`                               | :doc:`kolmogorov/crespi/full <pair_kolmogorov_crespi_full>` |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`kolmogorov/crespi/z <pair_kolmogorov_crespi_z>`        | :doc:`lcbop <pair_lcbop>`                                         | :doc:`lebedeva/z <pair_lebedeva_z>`                 | :doc:`lennard/mdf <pair_mdf>`                               |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`line/lj <pair_line_lj>`                                | :doc:`list <pair_list>`                                           | :doc:`lj/charmm/coul/charmm (iko) <pair_charmm>`    | :doc:`lj/charmm/coul/charmm/implicit (ko) <pair_charmm>`    |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`lj/charmm/coul/long (gikot) <pair_charmm>`             | :doc:`lj/charmm/coul/long/soft (o) <pair_fep_soft>`               | :doc:`lj/charmm/coul/msm (o) <pair_charmm>`         | :doc:`lj/charmmfsw/coul/charmmfsh <pair_charmm>`            |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`lj/charmmfsw/coul/long <pair_charmm>`                  | :doc:`lj/class2 (gko) <pair_class2>`                              | :doc:`lj/class2/coul/cut (ko) <pair_class2>`        | :doc:`lj/class2/coul/cut/soft <pair_fep_soft>`              |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`lj/class2/coul/long (gko) <pair_class2>`               | :doc:`lj/class2/coul/long/soft <pair_fep_soft>`                   | :doc:`lj/class2/soft <pair_fep_soft>`               | :doc:`lj/cubic (go) <pair_lj_cubic>`                        |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`lj/cut (gikot) <pair_lj>`                              | :doc:`lj/cut/coul/cut (gko) <pair_lj>`                            | :doc:`lj/cut/coul/cut/soft (o) <pair_fep_soft>`     | :doc:`lj/cut/coul/debye (gko) <pair_lj>`                    |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`lj/cut/coul/dsf (gko) <pair_lj>`                       | :doc:`lj/cut/coul/long (gikot) <pair_lj>`                         | :doc:`lj/cut/coul/long/cs <pair_cs>`                | :doc:`lj/cut/coul/long/soft (o) <pair_fep_soft>`            |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`lj/cut/coul/msm (go) <pair_lj>`                        | :doc:`lj/cut/coul/wolf (o) <pair_lj>`                             | :doc:`lj/cut/dipole/cut (go) <pair_dipole>`         | :doc:`lj/cut/dipole/long (g) <pair_dipole>`                 |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`lj/cut/dipole/sf (go) <pair_dipole>`                   | :doc:`lj/cut/soft (o) <pair_fep_soft>`                            | :doc:`lj/cut/thole/long (o) <pair_thole>`           | :doc:`lj/cut/tip4p/cut (o) <pair_lj>`                       |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`lj/cut/tip4p/long (ot) <pair_lj>`                      | :doc:`lj/cut/tip4p/long/soft (o) <pair_fep_soft>`                 | :doc:`lj/expand (gko) <pair_lj_expand>`             | :doc:`lj/expand/coul/long (g) <pair_lj_expand>`             |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`lj/gromacs (gko) <pair_gromacs>`                       | :doc:`lj/gromacs/coul/gromacs (ko) <pair_gromacs>`                | :doc:`lj/long/coul/long (iot) <pair_lj_long>`       | :doc:`lj/long/dipole/long <pair_dipole>`                    |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`lj/long/tip4p/long (o) <pair_lj_long>`                 | :doc:`lj/mdf <pair_mdf>`                                          | :doc:`lj/sdk (gko) <pair_sdk>`                      | :doc:`lj/sdk/coul/long (go) <pair_sdk>`                     |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`lj/sdk/coul/msm (o) <pair_sdk>`                        | :doc:`lj/sf/dipole/sf (go) <pair_dipole>`                         | :doc:`lj/smooth (o) <pair_lj_smooth>`               | :doc:`lj/smooth/linear (o) <pair_lj_smooth_linear>`         |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`lj/switch3/coulgauss/long <pair_lj_switch3_coulgauss>` | :doc:`lj96/cut (go) <pair_lj96>`                                  | :doc:`local/density <pair_local_density>`           | :doc:`lubricate (o) <pair_lubricate>`                       |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`lubricate/poly (o) <pair_lubricate>`                   | :doc:`lubricateU <pair_lubricateU>`                               | :doc:`lubricateU/poly <pair_lubricateU>`            | :doc:`mdpd <pair_meso>`                                     |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`mdpd/rhosum <pair_meso>`                               | :doc:`meam/c <pair_meamc>`                                        | :doc:`meam/spline (o) <pair_meam_spline>`           | :doc:`meam/sw/spline <pair_meam_sw_spline>`                 |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`mgpt <pair_mgpt>`                                      | :doc:`mie/cut (g) <pair_mie>`                                     | :doc:`momb <pair_momb>`                             | :doc:`morse (gkot) <pair_morse>`                            |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`morse/smooth/linear (o) <pair_morse>`                  | :doc:`morse/soft <pair_fep_soft>`                                 | :doc:`multi/lucy <pair_multi_lucy>`                 | :doc:`multi/lucy/rx (k) <pair_multi_lucy_rx>`               |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`nb3b/harmonic <pair_nb3b_harmonic>`                    | :doc:`nm/cut (o) <pair_nm>`                                       | :doc:`nm/cut/coul/cut (o) <pair_nm>`                | :doc:`nm/cut/coul/long (o) <pair_nm>`                       |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`oxdna/coaxstk <pair_oxdna>`                            | :doc:`oxdna/excv <pair_oxdna>`                                    | :doc:`oxdna/hbond <pair_oxdna>`                     | :doc:`oxdna/stk <pair_oxdna>`                               |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`oxdna/xstk <pair_oxdna>`                               | :doc:`oxdna2/coaxstk <pair_oxdna2>`                               | :doc:`oxdna2/dh <pair_oxdna2>`                      | :doc:`oxdna2/excv <pair_oxdna2>`                            |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`oxdna2/hbond <pair_oxdna2>`                            | :doc:`oxdna2/stk <pair_oxdna2>`                                   | :doc:`oxdna2/xstk <pair_oxdna2>`                    | :doc:`peri/eps <pair_peri>`                                 |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`peri/lps (o) <pair_peri>`                              | :doc:`peri/pmb (o) <pair_peri>`                                   | :doc:`peri/ves <pair_peri>`                         | :doc:`polymorphic <pair_polymorphic>`                       |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`python <pair_python>`                                  | :doc:`quip <pair_quip>`                                           | :doc:`reax/c (ko) <pair_reaxc>`                     | :doc:`rebo (io) <pair_airebo>`                              |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`resquared (go) <pair_resquared>`                       | :doc:`sdpd/taitwater/isothermal <pair_sdpd_taitwater_isothermal>` | :doc:`smd/hertz <pair_smd_hertz>`                   | :doc:`smd/tlsph <pair_smd_tlsph>`                           |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`smd/tri\_surface <pair_smd_triangulated_surface>`      | :doc:`smd/ulsph <pair_smd_ulsph>`                                 | :doc:`smtbq <pair_smtbq>`                           | :doc:`snap (k) <pair_snap>`                                 |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`snap (k) <pair_snap>`                                  | :doc:`soft (go) <pair_soft>`                                      | :doc:`sph/heatconduction <pair_sph_heatconduction>` | :doc:`sph/idealgas <pair_sph_idealgas>`                     |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`sph/lj <pair_sph_lj>`                                  | :doc:`sph/rhosum <pair_sph_rhosum>`                               | :doc:`sph/taitwater <pair_sph_taitwater>`           | :doc:`sph/taitwater/morris <pair_sph_taitwater_morris>`     |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`spin/dipole/cut <pair_spin_dipole>`                    | :doc:`spin/dipole/long <pair_spin_dipole>`                        | :doc:`spin/dmi <pair_spin_dmi>`                     | :doc:`spin/exchange <pair_spin_exchange>`                   |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`spin/magelec <pair_spin_magelec>`                      | :doc:`spin/neel <pair_spin_neel>`                                 | :doc:`srp <pair_srp>`                               | :doc:`sw (giko) <pair_sw>`                                  |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`table (gko) <pair_table>`                              | :doc:`table/rx (k) <pair_table_rx>`                               | :doc:`tdpd <pair_meso>`                             | :doc:`tersoff (giko) <pair_tersoff>`                        |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`tersoff/mod (gko) <pair_tersoff_mod>`                  | :doc:`tersoff/mod/c (o) <pair_tersoff_mod>`                       | :doc:`tersoff/table (o) <pair_tersoff>`             | :doc:`tersoff/zbl (gko) <pair_tersoff_zbl>`                 |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`thole <pair_thole>`                                    | :doc:`tip4p/cut (o) <pair_coul>`                                  | :doc:`tip4p/long (o) <pair_coul>`                   | :doc:`tip4p/long/soft (o) <pair_fep_soft>`                  |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`tri/lj <pair_tri_lj>`                                  | :doc:`ufm (got) <pair_ufm>`                                       | :doc:`vashishta (gko) <pair_vashishta>`             | :doc:`vashishta/table (o) <pair_vashishta>`                 |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+
| :doc:`yukawa (gko) <pair_yukawa>`                            | :doc:`yukawa/colloid (go) <pair_yukawa_colloid>`                  | :doc:`zbl (gko) <pair_zbl>`                         |                                                             |
+--------------------------------------------------------------+-------------------------------------------------------------------+-----------------------------------------------------+-------------------------------------------------------------+


.. _lws: http://lammps.sandia.gov
.. _ld: Manual.html
.. _lc: Commands_all.html