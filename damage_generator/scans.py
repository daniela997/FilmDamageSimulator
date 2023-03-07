import json
import os
import numpy as np
import cv2 as cv
import pandas as pd

from helpers import pad_to_size, points_to_contours, sorted_alphanumeric
from unit_converter import PixelToMicron

""" Load JSON artifact contours in a pandas dataframe. """

def df_contours(contours_list, artifact_type, img, unit_converter):
    images_df_list = []
    mask = np.zeros_like(img)

    #quadrant_mask = np.arange(70).reshape(7,-1)
    quadrant_mask = np.indices((14,20)).transpose(1,2,0)
    quadrant_mask = np.repeat(quadrant_mask, 256, axis=0)
    quadrant_mask = np.repeat(quadrant_mask, 256, axis=1)

    for cnt in contours_list:
        area = cv.contourArea(cnt)
        if area != 0:
            rect = cv.boundingRect(cnt)
            x,y,w,h = rect
            area_sq_mm = unit_converter.convert_area(area) * 1e-6
            area_micron = unit_converter.convert_area(area)
            cv.drawContours(mask, [cnt], -1, 255, -1)
            artifact = cv.bitwise_and(img[y:y+h, x:x+w], mask[y:y+h, x:x+w])
            M = cv.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            quadrant = quadrant_mask[cY, cX]
            images_df_list.append(
                                    {
                                        'Contour': cnt, 
                                        'Contour Area (px)': area, 
                                        'Contour Area (sq mm)': area_sq_mm, 
                                        'Contour Area (sq micron)': area_micron, 
                                        'bbox x': x, 
                                        'bbox y': y, 
                                        'bbox w': w, 
                                        'bbox h': h, 
                                        'centroid x': cX,
                                        'centroid y': cY,
                                        'Quandrant': tuple(quadrant), 
                                        'Min bounding bounding box (center (x,y), (w, h), angle)': cv.minAreaRect(cnt), 
                                        'Type': artifact_type,
                                        'Artifact': artifact,
                                    }
                                )
    images_df = pd.DataFrame(images_df_list)
    return images_df


""" Load scans and process annotated artifacts to dataframe """
def load_scans(scan_dir, verbose=True):
    df_all_artifacts_all_scans = pd.DataFrame()
    for img_filename in sorted_alphanumeric(os.listdir(scan_dir)):
        if img_filename.endswith(".jpg"):
            if verbose: print('Loading dust scan ', img_filename)
            img = cv.cvtColor(cv.imread(scan_dir+img_filename), cv.COLOR_BGR2GRAY)
            filename = os.path.splitext(img_filename)[0]
            json_filename = filename+'.json'

            with open(scan_dir+json_filename) as f:
                if verbose: print('Loading json annotations ', )
                d = json.load(f)

            long_hair_contours = []
            short_hair_contours = []
            scratch_contours = []
            dirt_contours = []
            dust_contours = []
            
            if verbose: print('Converting contours to opencv format...')
            for key, value in d.items():
                contours = points_to_contours(d[key]['points'])
                if filename in ['Scan (8)', 'Scan (9)', 'Scan (10)']:
                    contours[:, :, 0] = contours[:, :, 0] * 1.5
                    contours[:, :, 1] = contours[:, :,  1] * 1.5
                if d[key]['label']['name'] == 'Dust':
                    dust_contours.append(contours)
                elif d[key]['label']['name'] == 'Dirt':
                    dirt_contours.append(contours)
                elif d[key]['label']['name'] == 'Scratch':
                    scratch_contours.append(contours)
                elif d[key]['label']['name'] == 'Long hair':
                    long_hair_contours.append(contours)
                elif d[key]['label']['name'] == 'Short hair':
                    short_hair_contours.append(contours)

            if verbose: print('Creating padded artifacts...')
            unit_converter = PixelToMicron(img.shape[1], img.shape[0])

            dfs_list = []
            if len(dust_contours) > 0: 
                df_dust = df_contours(dust_contours, 'dust', img, unit_converter)
                pad_to_size(df_dust)
                dfs_list.append(df_dust)

            if len(dirt_contours) > 0: 
                df_dirt = df_contours(dirt_contours, 'dirt', img, unit_converter)
                pad_to_size(df_dirt)
                dfs_list.append(df_dirt)

            if len(scratch_contours) > 0: 
                df_scratch = df_contours(scratch_contours, 'scratch', img, unit_converter)
                pad_to_size(df_scratch)
                dfs_list.append(df_scratch)

            if len(long_hair_contours) > 0: 
                df_long_hair = df_contours(long_hair_contours, 'long hair', img, unit_converter)
                pad_to_size(df_long_hair)
                dfs_list.append(df_long_hair)

            if len(short_hair_contours) > 0: 
                df_short_hair = df_contours(short_hair_contours, 'short hair', img, unit_converter)
                pad_to_size(df_short_hair)
                dfs_list.append(df_short_hair)

            
            df_all_artifacts = pd.concat(dfs_list, axis=0, ignore_index=True)
            df_all_artifacts['Scan'] = filename
            if verbose: print('Extracted artifacts for ', filename)

            df_all_artifacts_all_scans = pd.concat([df_all_artifacts_all_scans, df_all_artifacts], axis=0, ignore_index=True)
    
    return df_all_artifacts_all_scans

def load_images(synthetic_artifacts_dir, artifact_type, verbose=False):
    images = []
    images_df_list = []
    artifact_map = {
        'stain' : 'dirt',
        'spots' : 'dirt',
        'lint' : 'short hair',
        'dirt' : 'dirt',
        'dots' : 'dirt',
        'scratches' : 'dirt',
        'hair-short' : 'short hair',
        'sprinkles' : 'dust',
        'hair' : 'long hair',
        'smut' : 'dirt',
    }
    for img_filename in sorted_alphanumeric(os.listdir(synthetic_artifacts_dir+'/'+artifact_type+'/')):
        if img_filename.endswith(".png"):
            if verbose: print('Loading image overlay ', img_filename)
            img = cv.imread(synthetic_artifacts_dir+'/'+artifact_type+'/'+img_filename, cv.IMREAD_UNCHANGED)[:,:,3]
            if img is not None:        
                #images.append(img)
                
                ret, thresh = cv.threshold(img, 127, 255, 0)
                kernel = np.ones((3,3), np.uint8)
                contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
                cnt = max(contours, key=len)
                images_df_list.append(
                                        {
                                            'Contour': cnt, 
                                            'Contour Area (px)': cv.contourArea(cnt), 
                                            'Non-zero pixel area': cv.countNonZero(img),
                                            'Bounding box (x, y, w, h)': cv.boundingRect(cnt), 
                                            'Min bounding bounding box (center (x,y), (w, h), angle)': cv.minAreaRect(cnt), 
                                            'Original Type': artifact_type,
                                            'Type': artifact_map[artifact_type],
                                            'Artifact': img,
                                        }
                                      )
    images_df = pd.DataFrame(images_df_list)
    return images_df#, images_np

def load_all_synthetic_images(synthetic_artifacts_dir, verbose=False):
    subdirs = next(os.walk(synthetic_artifacts_dir))[1]
    df_synthetic_artifacts = pd.DataFrame()
    for subdir in subdirs:
        df_synthetic_artifacts = pd.concat([df_synthetic_artifacts, 
                load_images(synthetic_artifacts_dir, subdir, verbose=verbose),
                ], axis=0, ignore_index=True)
    return df_synthetic_artifacts.sample(frac=1).reset_index(drop=True)