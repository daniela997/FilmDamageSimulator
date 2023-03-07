import numpy as np
import random
import pandas as pd
import math
import skimage.transform as skimage_tf
import cv2 as cv
from sample import sample_num_artifacts, sample_size_artifacts, sample_closest_in_area

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise

def increase_contrast(pixvals):
    minval = np.percentile(pixvals, 2)
    maxval = np.percentile(pixvals, 98)
    pixvals = np.clip(pixvals, minval, maxval)
    pixvals = ((pixvals - minval) / (maxval - minval))
    return pixvals

def random_perlin_with_numpy(num_samples, noise_array):
    # Create a flat copy of the array
    linear_idx = np.random.choice(noise_array.size, p=noise_array.ravel()/float(noise_array.sum()), size=num_samples)
    x, y = np.unravel_index(linear_idx, noise_array.shape)
    return x, y

def shift_bit_length(x):
    return 1<<(x-1).bit_length()

def line_scratch(length):
    length_pot = shift_bit_length(length.item())
    noise_scale = np.random.randint(1, 4, size=1, dtype=int)[0].item()
    perlin_noise = generate_perlin_noise_2d((length_pot, length_pot), (2**noise_scale, 2**noise_scale))
    normalised_noise = (perlin_noise - np.min(perlin_noise))/np.ptp(perlin_noise)
    slice_fade = np.random.randint(low=0, high=length-20, size=1, dtype=int)[0]
    fade = increase_contrast(normalised_noise)[0:length.item(), slice_fade.item():slice_fade.item()+20]
    fade *= (255.0/fade.max())
    mean = 0
    var = 15
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (fade.shape[0], fade.shape[1])) 
    fade = fade + gaussian
    fade = fade.astype(np.uint8)

    num_lines = np.random.randint(1, 2, size=1, dtype=int)
    lines_xs = np.random.randint(1, 19, size=num_lines, dtype=int)
    lines_ys = np.random.randint(0, int(length*0.2), size=num_lines, dtype=int)

    line_mask_width, line_mask_height = 20, length
    line_mask = np.ones((line_mask_height, line_mask_width)) * 255
    line_thickness = 1

    
    x1, y1 = lines_xs[0], lines_ys[0]
    x2, y2 = lines_xs[0], np.random.randint(lines_ys[0]+10, length-(lines_ys[0]+10), size=1, dtype=int)
    line = cv.line(line_mask, (x1, y1), (x2, int(y2)), (0, 255, 0), thickness=line_thickness).astype(np.uint8)

    scratch = cv.bitwise_and(fade, np.invert(line)).astype(np.uint8)
    scratch = cv.GaussianBlur(scratch, (3,3), 0).astype(np.uint8)
    scratch = cv.resize(scratch, None, fx = 0.5, fy = 1, interpolation = cv.INTER_CUBIC)

    return scratch



def create_random_mask(target_size, df, df_synthetic, df_counts,
                       use_synthetic=False, 
                       min_artifacts=5, max_artifacts=100, rescale=True, 
                       uniform_sample=False, verbose=True):
   
    if rescale:
        rescale_factor = target_size[0]/2560 if target_size[0] <= target_size[1] else target_size[1]/2560
    else:
        rescale_factor = 1.
    if not uniform_sample:

        num_dust = sample_num_artifacts(df_counts, 'dust', target_size)
        num_dirt = sample_num_artifacts(df_counts, 'dirt', target_size)
        num_long_hair = sample_num_artifacts(df_counts, 'long hair', target_size)
        num_short_hair = sample_num_artifacts(df_counts, 'short hair', target_size)
        num_scratch = sample_num_artifacts(df_counts, 'scratch', target_size)
        
        add_scratches = random.random()
        num_extra_scratch = np.random.gamma(6, 2, 1).astype(int) if add_scratches > 0.5 else 0

        resize_targets_dust = sample_size_artifacts(df, 'dust', num_dust)
        resize_targets_dirt = sample_size_artifacts(df, 'dirt', num_dirt)
        resize_targets_short_hair = sample_size_artifacts(df, 'short hair', num_short_hair)
        resize_targets_long_hair= sample_size_artifacts(df, 'long hair', num_long_hair)
        resize_targets_scratch = sample_size_artifacts(df, 'scratch', num_scratch)

        if verbose: print(
            "Sampling {} dust artifacts, {} dirt artifacts, {} long hairs, {} short hairs, {} scratches and {} extra scratches...".format(
                num_dust, num_dirt, num_long_hair, num_short_hair, num_scratch, num_extra_scratch
            ))
        
        if use_synthetic: 
            df_merged = pd.concat([df[['Type', 'Contour Area (px)', 'Artifact']].rename(columns={'Contour Area (px)':'Contour Area'}),
                               df_synthetic[['Type', 'Non-zero pixel area', 'Artifact']].rename(columns={'Non-zero pixel area':'Contour Area'})],
                              ignore_index=True)
            # df_merged = df_synthetic[['Type', 'Non-zero pixel area', 'Artifact']].rename(columns={'Non-zero pixel area':'Contour Area'})
        else:
            df_merged = df[['Type', 'Contour Area (px)', 'Artifact']].rename(columns={'Contour Area (px)':'Contour Area'})

        df_merged = df_merged.sample(frac=1).reset_index(drop=True)
        
        df1 = df_merged.loc[df_merged['Type'] == 'dust'].reset_index(drop=True)
        df2 = df_merged.loc[df_merged['Type'] == 'dirt'].reset_index(drop=True)
        df3 = df_merged.loc[df_merged['Type'] == 'short hair'].reset_index(drop=True)
        df4 = df_merged.loc[df_merged['Type'] == 'long hair'].reset_index(drop=True)
        df5 = df_merged.loc[df_merged['Type'] == 'scratch'].reset_index(drop=True)

        df1 = sample_closest_in_area(df1, resize_targets_dust)
        df2 = sample_closest_in_area(df2, resize_targets_dirt)
        df3 = sample_closest_in_area(df3, resize_targets_short_hair)
        df4 = sample_closest_in_area(df4, resize_targets_long_hair)
        df5 = sample_closest_in_area(df5, resize_targets_scratch)

        selected_artifacts_df = pd.concat([
                                           df1, 
                                           df2, 
                                           df3, 
                                           df4, 
                                           df5,
                                           ])

        artifacts_num = num_dust + num_dirt + num_short_hair + num_long_hair + num_scratch + num_extra_scratch


    else: 
        artifacts_num = np.random.randint(min_artifacts, max_artifacts)
        selected_artifacts_df = df.sample(artifacts_num)
        if verbose: print("Number of artifacts in generated mask: ", artifacts_num)

    mask_final = np.zeros(target_size).astype(np.uint8)

    perlin_noise = generate_perlin_noise_2d(target_size, (2, 2))
    normalised_noise = (perlin_noise - np.min(perlin_noise))/np.ptp(perlin_noise)
    contrast_perlin_noise = increase_contrast(normalised_noise)
    xs, ys = random_perlin_with_numpy(artifacts_num, normalised_noise)
    i = 0
    random_angles = np.random.randint(0, 360, size=artifacts_num)

    for _, artifact_row in selected_artifacts_df.iterrows():
        try:
            artifact_var_idx = np.random.randint(1, 4)
            artifact = artifact_row['Artifact'].astype(np.uint8)

            random_scale = artifact_row['Target size']/artifact_row['Contour Area']
            random_angle = random_angles[i]

            if verbose: print("Global rescale factor {}, Random scale {}, Product {}".format(
                rescale_factor, math.sqrt(random_scale), rescale_factor*math.sqrt(random_scale)))

            new_rescale_factor = rescale_factor * math.sqrt(random_scale)
            #new_rescale_factor = new_rescale_factor if new_rescale_factor > 0.05 else 0.1
            artifact = skimage_tf.rescale(artifact, round(new_rescale_factor, 2), anti_aliasing=True, preserve_range=True)
            artifact = skimage_tf.rotate(artifact, angle=random_angle, resize=True, preserve_range=True)
            artifact_w, artifact_h = artifact.shape[:2]

            x1 = xs[i] - artifact_w // 2
            x2 = x1 + artifact_w

            if x1 < 0:
                artifact = artifact[-x1:,:]
                x1 = 0
            if x2 > target_size[0]:
                artifact = artifact[:-(x2-target_size[0]), :]
                x2 = target_size[0]

            y1 = ys[i] - artifact_h // 2
            y2 = y1 + artifact_h
            
            if y1 < 0:
                artifact = artifact[:, -y1:]
                y1 = 0
            if y2 > target_size[1]:
                artifact = artifact[:, :-(y2-target_size[1])]
                y2 = target_size[1]

            if verbose:
                print("Random location centroid x: {}, y: {}.".format(xs[i], ys[i]))
                
            mask_final[x1:x2,y1:y2] = np.where(artifact>mask_final[x1:x2,y1:y2], artifact, mask_final[x1:x2,y1:y2])
            i=i+1

        except Exception:
            i=i+1
            pass
    
    if add_scratches > 0.5:
        extra_scratches_lengths = np.random.randint(10, high=target_size[0], size=num_extra_scratch, dtype=int)
        horizontal = random.random()
        random_angle_scratches = np.random.randint(0, 2, size=1)


        for scratch_length in extra_scratches_lengths:
            try: 
                scratch = line_scratch(scratch_length)
                scratch = skimage_tf.rotate(scratch, angle=random_angle_scratches, resize=True, preserve_range=True)

                if horizontal > 0.5 : scratch = np.rot90(scratch)
                scratch_w, scratch_h = scratch.shape[:2]

                x1 = xs[i] - scratch_w // 2
                x2 = x1 + scratch_w

                if x1 < 0:
                    scratch = scratch[-x1:,:]
                    x1 = 0
                if x2 > target_size[0]:
                    scratch = scratch[:-(x2-target_size[0]), :]
                    x2 = target_size[0]

                y1 = ys[i] - scratch_h // 2
                y2 = y1 + scratch_h
                
                if y1 < 0:
                    scratch = scratch[:, -y1:]
                    y1 = 0
                if y2 > target_size[1]:
                    scratch = scratch[:, :-(y2-target_size[1])]
                    y2 = target_size[1]

                if verbose:
                    print("Random location centroid x: {}, y: {}.".format(xs[i], ys[i]))
                    
                mask_final[x1:x2,y1:y2] = np.where(scratch>mask_final[x1:x2,y1:y2], scratch, mask_final[x1:x2,y1:y2])
                
                i=i+1
            
            except Exception:
                i=i+1
                pass

    mask_final = np.invert(mask_final.astype(np.uint8))

    maxval = 255
    thresh = 240

    binarised_mask_final = (mask_final > thresh) * maxval

    del selected_artifacts_df

    return mask_final.astype(np.uint8), binarised_mask_final.astype(np.uint8), contrast_perlin_noise