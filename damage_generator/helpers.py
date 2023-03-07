import numpy as np
import re

""" Process scan filenames in alphanumeric order """
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

""" Convert points to contours """
def points_to_contours(points):
    contours = []
    for point_entry in points:
        grid = np.asarray([(point_entry['x'], point_entry['y'])])
        contours.append(grid)
    contours = np.asarray(contours, dtype='int32')
    return contours

""" Pad artifact contour to square """
def pad_artifact(target_shape, artifact):
    padded = np.zeros(target_shape)
    w, h = artifact.shape
    padded[(target_shape[0]-w)//2:(target_shape[0]-w)//2+w, (target_shape[1]-h)//2:(target_shape[1]-h)//2+h] = artifact
    return padded

""" Pad all artifacts in dataframe to target size """
def pad_to_size(df):
    pad_to_size = df['bbox w'].max() if df['bbox w'].max() > df['bbox h'].max() else df['bbox h'].max()
    df["Padded Artifact"] = df["Artifact"].apply(lambda x: pad_artifact((pad_to_size, pad_to_size), x))