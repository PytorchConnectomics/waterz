import numpy as np
import mahotas
from scipy import ndimage
from tqdm import tqdm

def get_seeds(boundary, method='grid', next_id = 1,
             seed_distance = 10):
    if method == 'grid':
        height = boundary.shape[0]
        width  = boundary.shape[1]

        seed_positions = np.ogrid[0:height:seed_distance, 0:width:seed_distance]
        num_seeds_y = seed_positions[0].size
        num_seeds_x = seed_positions[1].size
        num_seeds = num_seeds_x*num_seeds_y
        seeds = np.zeros_like(boundary).astype(np.int32)
        seeds[seed_positions] = np.arange(next_id, next_id + num_seeds).reshape((num_seeds_y,num_seeds_x))

    if method == 'minima':
        minima = mahotas.regmin(boundary)
        seeds, num_seeds = mahotas.label(minima)
        seeds += next_id
        seeds[seeds==next_id] = 0

    if method == 'maxima_distance':
        distance = mahotas.distance(boundary<0.5)
        maxima = mahotas.regmax(distance)
        seeds, num_seeds = mahotas.label(maxima)
        seeds += next_id
        seeds[seeds==next_id] = 0
    return seeds, num_seeds

def watershed_concat(affs, affs_ran=255.0, seed_method='maxima_distance', use_mahotas_watershed = True):
    fragments = np.zeros(affs[0].shape).astype(np.uint64)
    next_id = 1
    for z in tqdm(range(affs.shape[1])):
        affs_z = affs[1:,z]
        if not isinstance(affs, np.ndarray):
            affs_z = np.array(affs_z)
        if affs_ran != 1:
            affs_z = affs_z / affs_ran
        boundary = 1 - affs_z.mean(axis=0)
        seeds, num_seeds = get_seeds(boundary, next_id=next_id, method=seed_method)
        if use_mahotas_watershed:
            fragments[z] = mahotas.cwatershed(boundary, seeds)
        else:
            fragments[z] = ndimage.watershed_ift((255*boundary).astype(np.uint8), seeds)
        next_id += num_seeds

    return fragments
