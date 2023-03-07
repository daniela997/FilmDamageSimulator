import numpy as np
import pandas as pd
import scipy.stats as stats
import random

def get_random_patch_of_size(img_size, target_size):
    if img_size[0]-target_size[0]-1 > 0 and img_size[1]-target_size[1]-1 > 0:
        x1 = np.random.randint(0, img_size[0]-target_size[0]-1)
        y1 = np.random.randint(0, img_size[1]-target_size[1]-1)

        x2, y2 = x1+target_size[0], y1+target_size[1]
    else: 
        x1, x2, y1, y2 = None, None, None, None
    return x1, x2, y1, y2

def sample_size_artifacts(df, artifact_type, num_artifact):
    min = df.loc[df['Type'] == artifact_type]['Contour Area (px)'].min()    
    max = df.loc[df['Type'] == artifact_type]['Contour Area (px)'].max()    
    x = np.linspace(min, max)
    gamma = stats.gamma
    gamma_param = gamma.fit(df.loc[df['Type'] == artifact_type]['Contour Area (px)'], floc=0)
    pdf_fitted = gamma.pdf(x, *gamma_param)
    shape, _, scale = gamma_param
    samples = np.random.gamma(shape, scale, num_artifact)
    return samples

def sample_num_artifacts(df_counts, artifact_type, target_size=None):
    ratio = target_size[0]/256
    min = df_counts.loc[df_counts['Type'] == artifact_type]['Counts'].min()
    max = df_counts.loc[df_counts['Type'] == artifact_type]['Counts'].max()
    # Sometimes we don't want any artifacts
    x = np.linspace(0, max)
    gamma = stats.gamma
    gamma_param = gamma.fit(df_counts.loc[df_counts['Type'] == artifact_type]['Counts'], floc=0)
    pdf_fitted = gamma.pdf(x, *gamma_param)
    shape, _, scale = gamma_param
    num_artifact = np.random.gamma(shape, scale, 1)
    #num_artifact = num_artifact * (ratio**2)
    return num_artifact.astype(int)

def sample_closest_in_area(df, target_areas):
    df = df.sample(frac=1).reset_index(drop=True) # reshuffle
    dust_areas = df['Contour Area']
    indexes = []
    for i in target_areas:
        df_sort = df.iloc[(dust_areas-i).abs().argsort()[:15]]
        candidate_indexes = df_sort.index.tolist()
        index = random.choice(candidate_indexes)
        indexes.append(index)
        dust_areas = dust_areas.drop(dust_areas.index[[index]])
    df = df.iloc[indexes].copy()
    df['Target size'] = target_areas
    return df