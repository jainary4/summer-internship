#!/usr/bin/env python
"""                                                                            
Algorithm that identifies the largest possible self-gravitating iso-density contours
of a certain particle type. This newer version relies on the load_from_snapshot routine from GIZMO.

Usage: CloudPhinder.py <snapshots> ... [options]

Options:                                                                       
   -h --help                  Show this screen.

   --outputfolder=<name>      Specifies the folder to save the outputs to, None defaults to the same location as the snapshot [default: None]
   --ptype=<N>                GIZMO particle type to analyze [default: 0]
   --G=<G>                    Gravitational constant to use; should be consistent with what was used in the simulation. [default: 4.301e4]
   --cluster_ngb=<N>          Length of particle's neighbour list. [default: 32]
   --nmin=<n>                 Minimum H number density to cut at, in cm^-3 [default: 1]
   --softening=<L>            Force softening for potential, if species does not have adaptive softening. [default: 1e-5]
   --fuzz=<L>                 Randomly perturb particle positions by this small fraction to avoid problems with particles at the same position in 32bit floating point precision data [default: 0]
   --alpha_crit=<f>           Critical virial parameter to be considered bound [default: 2.]
   --np=<N>                   Number of snapshots to run in parallel [default: 1]
   --ntree=<N>                Number of particles in a group before a BH-tree should be constructed to compute its properties [default: 1000]
   --overwrite                Whether to overwrite pre-existing clouds files
   --units_already_physical   Whether to convert units to physical from comoving
   --starforge_units 
   --max_linking_length=<L>   Maximum radius for neighbor search around a particle [default: 1e100]
"""

## from builtin
from os import path

import numpy as np

from docopt import docopt

from multiprocessing import Pool
import itertools

## from here
import sys
sys.path.append('./')
from cloudphinder.io_tools import (
    parse_filepath,
    read_particle_data,
    parse_particle_data,
    computeAndDump,
    SaveArrayDict,
)
from cloudphinder.clump_tools import ComputeGroups


def CloudPhind(filepath, options, particle_data=None, loud=True):
    """
    Input:
    filepath - path to snapshot data, used to determine output filename
        so it is not optional when particle_data is not None.
    options - CLI arguments defined by CloudPhinder.__doc__
    particle_data=None - pre-loaded particle data in a dictionary matching
        GIZMO keys.
    loud=True - flag to print to the console
    """

    ## parses filepath and reformats outputfolder if necessary
    snapnum, snapdir, snapname, outputfolder = parse_filepath(
        filepath, options["--outputfolder"]
    )

    ## skip if the file was not parseable
    if snapnum is None:
        return False

    ## generate output filenames
    nmin = float(options["--nmin"])
    alpha_crit = float(options["--alpha_crit"])

    hdf5_outfilename = (
        outputfolder + "/" + "Clouds_%s_n%g_alpha%g.hdf5" % (snapnum, nmin, alpha_crit)
    )
    dat_outfilename = (
        outputfolder + "/" + "bound_%s_n%g_alpha%g.dat" % (snapnum, nmin, alpha_crit)
    )

    ## check if output already exists, if we aren't being asked to overwrite, short circuit
    overwrite = options["--overwrite"]
    if path.isfile(dat_outfilename) and not overwrite:
        if loud:
            print("File already exists and --overwrite=False, exiting.")
        return False

    ## read particle data from disk and apply dense gas cut
    ##  also unpacks relevant variables
    ptype = int(options["--ptype"])
    cluster_ngb = int(float(options["--cluster_ngb"]) + 0.5)

    ## load the particle data from disk if we weren't provided any
    if particle_data is None:
        particle_data = read_particle_data(
            snapnum,
            snapdir,
            snapname,
            ptype,
            cluster_ngb,
            softening=float(options["--softening"]),
            units_already_physical=bool(options["--units_already_physical"]),
        )

        ## skip this snapshot, there probably weren't enough particles
        if particle_data is None:
            return False

    ## unpack the particle data
    (new_particle_data, x, m, rho, hsml, u, v, zz, sfr) = parse_particle_data(
        particle_data, nmin, cluster_ngb
    )
    phi = np.zeros_like(rho)

    if new_particle_data is None:
        return False
    print(v.shape, v[:, 0], v[:, 0].std())
    ## call the cloud finder itself
    groups, bound_groups, assigned_groups = ComputeGroups(
        x,
        m,
        rho,
        phi,
        hsml,
        u,
        v,
        zz,
        sfr,
        cluster_ngb=cluster_ngb,
        max_linking_length=float(options["--max_linking_length"]),
        nmin=nmin,
        ntree=int(options["--ntree"]),
        alpha_crit=alpha_crit,
    )

    ## compute some basic properties of the clouds and dump them and
    ##  the particle data to disk
    computeAndDump(
        x,
        m,
        hsml,
        v,
        u,
        new_particle_data,
        ptype,
        bound_groups,
        hdf5_outfilename,
        dat_outfilename,
        overwrite,
    )

    return True


def main(options):

    nproc = int(options["--np"])

    snappaths = [p for p in options["<snapshots>"]]
    if nproc == 1:
        for f in snappaths:
            CloudPhind(f, options)
    else:
        argss = zip(snappaths, itertools.repeat(options))
        with Pool(nproc) as my_pool:
            my_pool.starmap(CloudPhind, argss, chunksize=1)


if __name__ == "__main__":
    options = docopt(__doc__)
    main(options)
