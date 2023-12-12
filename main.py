import yaml
import cv2
import open3d as o3d
import numpy as np

from kitti_video import KittiVideo
from kitti_caliberation import KittiCalibration
from maps import OccupancySemanticMap

if __name__ == "__main__":
    
    with open('config/settings.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # occupancy semantic map
    osm = OccupancySemanticMap(config)
    
    # kitti scene parser
    kitti_scene = KittiVideo(
        video_root=config['KITTI']['video_root'],
        calib_root=config['KITTI']['calib_root']
    )

    # iteratively read the image and pointcloud from the kitti dataset
    for i in range(len(kitti_scene)):
        image, pointcloud, calib = kitti_scene[i]
        
        osm.update(image, pointcloud, calib)
            
        img_map = osm.get_map()
        img_segmetation = osm.get_segmentation()
        
        image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2), interpolation=cv2.INTER_NEAREST)
        img_segmetation = cv2.resize(img_segmetation, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # stack images
        image_stacked = np.hstack((image, img_segmetation))
        image_stacked = cv2.resize(image_stacked, (800, 187), interpolation=cv2.INTER_NEAREST)
        image_stacked = np.vstack((img_map, image_stacked))
        
        image_stacked  = cv2.resize(image_stacked, (800, 988), interpolation=cv2.INTER_NEAREST)

        cv2.imshow('Stacked', image_stacked)
        if cv2.waitKey(1) == ord('q'):
            break
        