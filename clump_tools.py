## from builtin
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from numba import njit, prange
from time import time

import numpy as np

## from github/mikegrudic

import pytreegrav as pytreegrav

## global variables
potential_mode = False
G_codeunits = 4.301e4
FRAC_AVIR_CHECK = 1e-4
TREEFORCE_THETA = 0.5


######## Energy functions ########
def KE(c, x, m, h, v, u):
    """c - clump index
    x - 2D array of positions (Nclump,N_this_clump)
    m - 2D array of masses (Nclump,N_this_clump)
    h - 2D array of smoothing lengths (Nclump,N_this_clump)
    v - 2D array of velocities (Nclump,N_this_clump)
    u - 2D array of internal energies (Nclump,N_this_clump)
    """
    mc, vc = m[c], v[c]

    ## velocity w.r.t. com velocity of clump
    v_well = vc - np.average(vc, weights=mc, axis=0)
    vSqr = np.sum(v_well**2, axis=1)
    return (mc * (vSqr / 2 + u[c])).sum()


def PE(c, x, m, h, v, u):
    phic = pytreegrav.Potential(
        x[c], m[c], h[c], G=G_codeunits, theta=TREEFORCE_THETA, parallel=True
    )
    return 0.5 * (phic * m[c]).sum()


def InteractionEnergy(
    x,
    m,
    h,
    group_a,
    tree_a,
    particles_not_in_tree_a,
    group_b,
):

    xb, mb, hb = x[group_b], m[group_b], h[group_b]
    num_in_b = len(group_b)
    if tree_a:
        ## evaluate potential from the particles in the tree
        # print(len(group_a))
        parallel = num_in_b > 100
        phi = pytreegrav.PotentialTarget(
            xb,
            None,  ## pos source
            None,  ## mass source
            hb,
            tree=tree_a,  ## source tree
            G=G_codeunits,
            theta=TREEFORCE_THETA,
            parallel=parallel,
        )

        ## brute force the particles not in the tree
        xa, ma, ha = (
            np.take(x, particles_not_in_tree_a, axis=0),
            np.take(m, particles_not_in_tree_a, axis=0),
            np.take(h, particles_not_in_tree_a, axis=0),
        )
        phi += pytreegrav.PotentialTarget(
            xb, xa, ma, hb, ha, G=G_codeunits, parallel=parallel
        )
    else:
        ## have to brute force all the particles
        xa, ma, ha = x[group_a], m[group_a], h[group_a]
        phi = pytreegrav.PotentialTarget(
            xb, xa, ma, hb, ha, G=G_codeunits, parallel=True
        )
    potential_energy = (mb * phi).sum()
    return potential_energy


def VirialParameter(c, x, m, h, v, u):
    ke, pe = KE(c, x, m, h, v, u), PE(c, x, m, h, v, u)
    return np.abs(2 * ke / pe)


######## Energy increment functions ########
def EnergyIncrement(
    i, c, m, M, x, v, u, h, v_com, tree=None, particles_not_in_tree=None
):

    phi = 0.0
    xtarget = np.array([x[i]])
    htarget = np.array([h[i]])
    if particles_not_in_tree:
        ## have to get potential from particles not in the tree by brute force
        xa, ma, ha = (
            np.take(x, particles_not_in_tree, axis=0),
            np.take(m, particles_not_in_tree, axis=0),
            np.take(h, particles_not_in_tree, axis=0),
        )
        phi += pytreegrav.PotentialTarget(
            xtarget, xa, ma, htarget, ha, G=G_codeunits  # , parallel=True
        )[0]
    if tree:
        phi += pytreegrav.PotentialTarget(
            xtarget,
            None,
            None,
            htarget,
            None,
            tree=tree,
            theta=TREEFORCE_THETA,
            G=G_codeunits,
            # parallel=True,  ## source pos and mass
        )

    vSqr = np.sum((v[i] - v_com) ** 2)
    mu = m[i] * M / (m[i] + M)
    return 0.5 * mu * vSqr + m[i] * u[i] + m[i] * phi


def KE_Increment(i, m, v, u, v_com, mtot):
    vSqr = np.sum((v[i] - v_com) ** 2)
    mu = m[i] * mtot / (m[i] + mtot)
    return 0.5 * mu * vSqr + m[i] * u[i]


def PE_Increment(i, c, m, x, v, u, v_com):
    phi = -G_codeunits * np.sum(m[c] / cdist([x[i]], x[c]))
    return m[i] * phi


######## Grouping functions ########
def ParticleGroups(
    x, m, rho, phi, h, u, v, ntree, alpha_crit, cluster_ngb=32, rmax=1e100
):

    if not potential_mode:
        phi = -rho
    ngbdist, ngb = cKDTree(x).query(
        x, min(cluster_ngb, len(x)), distance_upper_bound=min(rmax, h.max())
    )

    max_group_size = 0
    groups = {}
    particles_since_last_tree = {}
    group_tree = {}
    group_energy = {}
    group_KE = {}
    COM = {}
    v_COM = {}
    masses = {}
    bound_groups = {}
    assigned_group = -np.ones(len(x), dtype=np.int32)
    assigned_bound_group = -np.ones(len(x), dtype=np.int32)
    largest_assigned_group = -np.ones(len(x), dtype=np.int32)

    for i, _ in enumerate(x):
        ## do it one particle at a time, in decreasing order of density
        if not i % 10000:
            print(
                "Processed %d of %g particles; ~%3.2g%% done."
                % (i, len(x), 100 * (float(i) / len(x)) ** 2)
            )
        if np.any(ngb[i] > len(x) - 1):
            groups[i] = [i]
            group_tree[i] = None
            assigned_group[i] = i
            group_energy[i] = m[i] * u[i]
            group_KE[i] = m[i] * u[i]
            v_COM[i] = v[i]
            COM[i] = x[i]
            masses[i] = m[i]
            particles_since_last_tree[i] = [i]
            continue
        ngbi = ngb[i][1:]

        lower = phi[ngbi] < phi[i]
        if lower.sum():
            ngb_lower, ngbdist_lower = ngbi[lower], ngbdist[i][1:][lower]
            ngb_lower = ngb_lower[ngbdist_lower.argsort()]
            nlower = len(ngb_lower)
        else:
            nlower = 0

        add_to_existing_group = False
        if (
            nlower == 0
        ):  # if this is the densest particle in the kernel, let's create our own group with blackjack and hookers
            groups[i] = [i]
            group_tree[i] = None
            assigned_group[i] = i
            group_energy[i] = (
                m[i] * u[i]
            )  # - 2.8*m[i]**2/h[i] / 2 # kinetic + potential energy
            group_KE[i] = m[i] * u[i]
            v_COM[i] = v[i]
            COM[i] = x[i]
            masses[i] = m[i]
            particles_since_last_tree[i] = [i]
        # if there is only one denser particle, or both of the nearest two denser ones belong to the same group, we belong to that group too
        elif (
            nlower == 1 or assigned_group[ngb_lower[0]] == assigned_group[ngb_lower[1]]
        ):
            assigned_group[i] = assigned_group[ngb_lower[0]]
            groups[assigned_group[i]].append(i)
            add_to_existing_group = True
        # o fuck we're at a saddle point, let's consider both respective groups
        else:
            a, b = ngb_lower[:2]
            group_index_a, group_index_b = assigned_group[a], assigned_group[b]
            # make sure group a is the bigger one, switching labels if needed
            if masses[group_index_a] < masses[group_index_b]:
                group_index_a, group_index_b = group_index_b, group_index_a

            # if both dense boyes belong to the same group, that's the group for us too
            if group_index_a == group_index_b:
                assigned_group[i] = group_index_a
            # OK, we're at a saddle point, so we need to merge those groups
            else:
                group_a, group_b = groups[group_index_a], groups[group_index_b]
                ma, mb = masses[group_index_a], masses[group_index_b]
                xa, xb = COM[group_index_a], COM[group_index_b]
                va, vb = v_COM[group_index_a], v_COM[group_index_b]
                group_ab = group_a + group_b
                groups[group_index_a] = group_ab

                group_energy[group_index_a] += group_energy[group_index_b]
                group_KE[group_index_a] += group_KE[group_index_b]
                group_energy[group_index_a] += (
                    0.5 * ma * mb / (ma + mb) * np.sum((va - vb) ** 2)
                )  # energy due to relative motion: 1/2 * mu * dv^2
                group_KE[group_index_a] += (
                    0.5 * ma * mb / (ma + mb) * np.sum((va - vb) ** 2)
                )

                # mutual interaction energy; we've already counted their individual binding energies
                group_energy[group_index_a] += InteractionEnergy(
                    x,
                    m,
                    h,
                    group_a,
                    group_tree[group_index_a],
                    particles_since_last_tree[group_index_a],
                    group_b,
                )

                # we've got a big group, so we should probably do stuff with the tree
                if len(group_a) > ntree:
                    # if the smaller of the two is also large, let's build a whole new tree, and a whole new adventure
                    if len(group_b) > 512:
                        group_tree[group_index_a] = pytreegrav.ConstructTree(
                            np.take(x, group_ab, axis=0),
                            np.take(m, group_ab),
                            np.take(h, group_ab),
                        )
                        particles_since_last_tree[group_index_a][:] = []
                    # otherwise we want to keep the old tree from group a, and just add group b to the list of particles_since_last_tree
                    else:
                        particles_since_last_tree[group_index_a] += group_b
                else:
                    particles_since_last_tree[group_index_a][:] = group_ab[:]

                if len(particles_since_last_tree[group_index_a]) > ntree:
                    group_tree[group_index_a] = pytreegrav.ConstructTree(
                        np.take(x, group_ab, axis=0),
                        np.take(m, group_ab),
                        np.take(h, group_ab),
                    )

                    particles_since_last_tree[group_index_a][:] = []

                COM[group_index_a] = (ma * xa + mb * xb) / (ma + mb)
                v_COM[group_index_a] = (ma * va + mb * vb) / (ma + mb)
                masses[group_index_a] = ma + mb
                groups.pop(group_index_b, None)
                assigned_group[i] = group_index_a
                assigned_group[assigned_group == group_index_b] = group_index_a

                # if this new group is bound, we can delete the old bound group
                avir = abs(
                    2
                    * group_KE[group_index_a]
                    / np.abs(group_energy[group_index_a] - group_KE[group_index_a])
                )
                if np.random.rand() < FRAC_AVIR_CHECK:
                    R = avir / VirialParameter(group_ab, x, m, h, v, u)
                    if abs(np.log(R)) > 0.2:
                        print(
                            "1",
                            avir,
                            VirialParameter(group_ab, x, m, h, v, u),
                            group_KE[group_index_a],
                            KE(group_ab, x, m, h, v, u),
                            group_energy[group_index_a] - group_KE[group_index_a],
                            PE(group_ab, x, m, h, v, u),
                        )
                        print("Large error found in virial parameter! Exiting...")
                        exit()
                if avir < alpha_crit:
                    largest_assigned_group[group_ab] = len(group_ab)
                    assigned_bound_group[group_ab] = group_index_a

                for d in (
                    groups,
                    particles_since_last_tree,
                    group_tree,
                    group_energy,
                    group_KE,
                    COM,
                    v_COM,
                    masses,
                ):  # delete the data from the absorbed group
                    d.pop(group_index_b, None)
                add_to_existing_group = True

            groups[group_index_a].append(i)
            max_group_size = max(max_group_size, len(groups[group_index_a]))

        # assuming we've added a particle to an existing group, we have to update stuff
        if add_to_existing_group:
            g = assigned_group[i]
            mgroup = masses[g]
            group_KE[g] += KE_Increment(i, m, v, u, v_COM[g], mgroup)
            group_energy[g] += EnergyIncrement(
                i,
                groups[g][:-1],
                m,
                mgroup,
                x,
                v,
                u,
                h,
                v_COM[g],
                group_tree[g],
                particles_since_last_tree[g],
            )
            avir = abs(2 * group_KE[g] / np.abs(group_energy[g] - group_KE[g]))
            if np.random.rand() < FRAC_AVIR_CHECK:
                R = avir / VirialParameter(groups[g], x, m, h, v, u)
                if abs(np.log(R)) > 1:
                    print(
                        "2",
                        avir,
                        VirialParameter(groups[g], x, m, h, v, u),
                        group_KE[g],
                        KE(groups[g], x, m, h, v, u),
                        group_energy[g] - group_KE[g],
                        PE(groups[g], x, m, h, v, u),
                    )
                    print("Large error found in virial parameter! Exiting...")
                    exit()
            if avir < alpha_crit:
                largest_assigned_group[i] = len(groups[g])
                assigned_bound_group[groups[g]] = g

            v_COM[g] = (m[i] * v[i] + mgroup * v_COM[g]) / (m[i] + mgroup)
            masses[g] += m[i]
            particles_since_last_tree[g].append(i)
            if len(particles_since_last_tree[g]) > ntree:
                group_tree[g] = pytreegrav.ConstructTree(
                    x[groups[g]], m[groups[g]], h[groups[g]]
                )
                particles_since_last_tree[g][:] = []
            max_group_size = max(max_group_size, len(groups[g]))

    # Now assign particles to their respective bound groups
    # print((assigned_bound_group == -1).sum() / len(assigned_bound_group))
    for i, ai in enumerate(assigned_bound_group):
        # ai = assigned_bound_group[i]
        if ai < 0:
            continue

        if ai in bound_groups:
            bound_groups[ai].append(i)
        else:
            bound_groups[ai] = [i]

    return groups, bound_groups, assigned_group


def ComputeGroups(
    x,
    m,
    rho,
    phi,
    hsml,
    u,
    v,
    zz,
    sfr,
    cluster_ngb,
    max_linking_length,
    nmin,
    ntree,
    alpha_crit,
):

    ## cast arrays to double precision
    (x, m, rho, phi, hsml, u, v) = (
        np.float64(x),
        np.float64(m),
        np.float64(rho),
        np.float64(phi),
        np.float64(hsml),
        np.float64(u),
        np.float64(v),
    )

    # make sure no two particles are at the same position
    while len(np.unique(x, axis=0)) < len(x):
        x *= 1 + np.random.normal(size=x.shape) * 1e-8

    t = time()
    groups, bound_groups, assigned_group = ParticleGroups(
        x,
        m,
        rho,
        phi,
        hsml,
        u,
        v,
        ntree=ntree,
        alpha_crit=alpha_crit,
        cluster_ngb=cluster_ngb,
        rmax=max_linking_length,
    )
    t = time() - t
    print("Time: %g" % t)

    return groups, bound_groups, assigned_group