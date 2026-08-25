import pandas as pd
from PIL import Image
import os

from .base_dataset import GazeFollow_SocialGaze

class DyadicGaze(GazeFollow_SocialGaze):
    dataset_name = 'dyadic'

    def _load_data(self, annotation_csv):
        df = pd.read_csv(annotation_csv)
        df = df[['path', 'personID', 'hbox_xmin', 'hbox_ymin', 'hbox_xmax', 'hbox_ymax', 'gaze_x', 'gaze_y', 'inframe', 'gaze_pattern_id']]
        grouped_df = df.groupby(['path'])
        keys = list(grouped_df.groups.keys())
        return grouped_df, keys

    def _get_info(self, index):
        g = self.data.get_group(self.keys[index])
        info = {}

        # Initialize the lists as part of the 'info' dictionary
        info['normalized_hboxes'] = []
        info['normalized_gazes'] = []
        info['raw_hboxes'] = []
        info['raw_gazes'] = []
        info['gaze_vector'] = []
        info['inout_ls'] = []
        info['pattern_ls'] = []

        # Iterate through rows and populate the lists
        for _, row in g.iterrows():
            path, personID, xmin, ymin, xmax, ymax, gaze_x, gaze_y, inout, pattern = row
            info['normalized_hboxes'].append((xmin, ymin, xmax, ymax))
            info['normalized_gazes'].append((gaze_x, gaze_y))
            info['inout_ls'].append(inout)
            info['pattern_ls'].append(pattern)

        # Load the image and add it to the info dictionary
        raw_img = Image.open(os.path.join(self.frame_dir, path))
        info['path'] = path
        info['raw_img'] = raw_img.convert("RGB")

        # Get and store the image size (width and height)
        width, height = raw_img.size
        info['width'] = width
        info['height'] = height

        # Create the headboxes with raw coordinates
        for idx in range(len(info['normalized_hboxes'])):
            xmin, ymin, xmax, ymax = info['normalized_hboxes'][idx]
            xmin, ymin, xmax, ymax = map(float, [xmin*width, ymin*height, xmax*width, ymax*height])
            info['raw_hboxes'].append((xmin, ymin, xmax, ymax))

        return info