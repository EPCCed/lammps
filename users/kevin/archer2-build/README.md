# Building CGAL dependencies

This is all assumed to be in `PrgEnv-gnu`. A copy/link of these files
will be required in the root of the build (which should not be here).

## Ball pivoting

See the script `build-bpa.sh` for further details. Arrange the script
and the `CMakelists.txt` in a suitable location and run
```
# module load PrgEnv-gnu
$ bash ./build-bpa.sh`
```

This should result in directories `BallPivoting/include` and
`BallPivoting/lib` with library `libballpivoting.a`.

## CGAL

See the script `build-cgal.sh` for further deatils. Run
```
$ bash ./build-cgal.sh`
```

This should result in relevatn directories `cgal/include` and `cgal/lib`.

## LAMMPS

The main point here is that the `cmake` route seems to be extremely
painful (it doesn't work and it's very difficult to see why).

However, the older `Makefile` route is much more understandable.
So I've provided a `Makefile.kevin` which should go in `src/MAKE`.
There are some paths at the start of the `Makefile` which will
need adjusting to suit the local circumstances.

Specifically,
```
MY_BALL_PIVOTING=$(MY_ROOT)/BallPivoting
MY_CGAL=$(MY_ROOT)/cgal
```
should reflect the install location used above. The location of the
boost headers is also required.
