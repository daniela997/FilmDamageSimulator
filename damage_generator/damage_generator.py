from scans import load_scans, load_all_synthetic_images
from generate_masks import create_random_mask
import os
import argparse
import cv2 as cv
import uuid

parser = argparse.ArgumentParser(description='Generate film damage overlays')
parser.add_argument('--height', type=int, nargs='?', const=1024,
                    default=1024, help='height of the mask to be generated')

parser.add_argument('--width', type=int, nargs='?', const=1024,
                    default=1024, help='width of the mask to be generated')

parser.add_argument('--synthetic', action='store_true', help='use additional synthetic damage')
parser.set_defaults(synthetic=False)

parser.add_argument('--rescale', action='store_true', help='rescale artefatcs to match target resolution')
parser.set_defaults(rescale=True)

parser.add_argument('--binarised', action='store_true', help='binarise generated mask')
parser.set_defaults(binarised=True)

parser.add_argument('--verbose', action='store_true')
parser.set_defaults(verbose=False)

parser.add_argument('--uniform', action='store_true', help='uniform sampling instead of using fitted Gamma distributions')
parser.set_defaults(uniform=False)

args = parser.parse_args()

""" 1. Extract real artifacts from annotations """
scans_path = '/scans/'
synthetic_path = '/synthetic/'
abs_path = os.path.abspath(os.path.dirname(__file__))
scans_path = os.path.dirname(os.path.normpath(abs_path)) + scans_path
synthetic_path = os.path.dirname(os.path.normpath(abs_path)) + synthetic_path
df_artifacts = load_scans(scans_path, verbose=args.verbose)

""" 1.1 Optionally use synthetic artifacts """
if args.synthetic: 
    df_synthetic = df_synthetic_artifacts = load_all_synthetic_images(synthetic_path)
else: df_synthetic = None
df_per_patch_counts = df_artifacts.groupby(['Quandrant', 'Type']).size().to_frame('Counts').reset_index()

"""2. Generate mask of target size """
mask, binary_mask, perlin_noise = create_random_mask((args.height, args.width), df_artifacts, df_synthetic, df_per_patch_counts, use_synthetic=args.synthetic, rescale=args.rescale, uniform_sample=args.uniform, verbose=args.verbose)

directory = "/generated/"
directory = os.path.dirname(os.path.normpath(abs_path)) + directory

if not os.path.exists(directory):
    os.makedirs(directory)
    
cv.imwrite(directory+'mask_{}.png'.format(str(uuid.uuid4())), mask)

if args.binarised:
    cv.imwrite(directory+'binarised_mask_{}.png'.format(str(uuid.uuid4())), binary_mask)

print("Generated damage mask in {}".format(directory))
