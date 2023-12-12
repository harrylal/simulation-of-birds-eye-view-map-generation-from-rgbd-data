import numpy as np 
import cv2
from utils.lidar import clip_pointcloud
from pointpainting import PointPainter
from models.BiSeNetv2.model.BiseNetv2 import BiSeNetV2
from models.BiSeNetv2.utils.utils import preprocessing_kitti, postprocessing
from models.BiSeNetv2.utils.label import trainId2label
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import time
import torch

class OccupancySemanticMap:
    
    def __init__(self, config):
        
        self.config = config
        self.grid_w, self.grid_h = config['grid_mapping']['grid_size']
        self.grid_resx, self.grid_resy = config['grid_mapping']['grid_resolution']
        
        # Initialize the gray map
        self.map = np.zeros((self.grid_w, self.grid_h, 3), dtype=np.uint8)
        self.prev_pcd = None
        
        # Initialize the pointPainter
        self.paint = PointPainter()
        
        # clustering
        self.dbscan = DBSCAN(eps=1, min_samples=3)
        
        # Intialize the segmentation model 
        self.model = BiSeNetV2()
        self.checkpoint = torch.load(config['models']['segmentation']['model_weights'], map_location="cuda")
        self.model.load_state_dict(self.checkpoint['bisenetv2'], strict=False)        
        self.model.eval()
        self.model.cuda()  

        self.lidar_to_map = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]])

    def semantic_to_color(self, semantic):
        
        r = np.zeros((semantic.shape[:2])).astype(np.uint8)
        g = np.zeros((semantic.shape[:2])).astype(np.uint8)
        b = np.zeros((semantic.shape[:2])).astype(np.uint8)

        for key in trainId2label:
            label = trainId2label[key]
            if key == 255 or key == -1:
                continue
            id = key
            color = label.color
            indices = semantic == id
            r[indices], g[indices], b[indices] = color

        semantic = np.stack([b, g, r], axis=2)
        return semantic
    
    
    def lidar2map(self, pointcloud):
        """
        Convert pointcloud to lidar bev map
        Arguments:
            pointcloud: shape = [n_points, 4]
        Return:
            grid map
        """

        points_map = pointcloud[:,:3].copy()
        points_map = np.dot(points_map, self.lidar_to_map.T)

        x_map = points_map[:, 0]
        y_map = points_map[:, 1]
        z_map = points_map[:, 2]
        s = pointcloud[:, 3]  # semantic class IDs
        
        # Get the grid indices
        x_mapimg = np.floor(x_map / self.grid_resx).astype(np.int32)
        y_mapimg = np.floor(y_map / self.grid_resy).astype(np.int32)

        # Convert indices to image coordinates
        x_mapimg += int(self.grid_w / 2)
        y_mapimg += int(self.grid_h / 2)

        
        lidar_bev = np.zeros((self.grid_w, self.grid_h, 3), dtype=np.uint8)
        lidar_bev[:,:,:] = 255

        # Mark Ego Vehicle 
        cv2.drawMarker(lidar_bev, (int(self.grid_w / 2), int(self.grid_h / 2)), (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 20, 2)

        for i, point_class in enumerate(s):
            if point_class in trainId2label:
                label = trainId2label[point_class]
                color = label.color
                if 0 <= x_mapimg[i] < self.grid_w and 0 <= y_mapimg[i] < self.grid_h:
                    lidar_bev[x_mapimg[i], y_mapimg[i]] = color[::-1]

        return lidar_bev

    def append_cluster_centroids(self, point_cloud):

        # Get unique cluster IDs from the last column of the point cloud array
        cluster_ids = np.unique(point_cloud[:, -1])

        # Calculate centroids for each cluster
        centroids = []
        for cluster_id in cluster_ids:
            # Select points belonging to the current cluster
            cluster_points = point_cloud[point_cloud[:, -1] == cluster_id][:, :-2]  # Exclude the last column (cluster ID) and semantic id
            
            # Calculate centroid for the current cluster
            centroid = np.mean(cluster_points, axis=0)
            
            # Assign the cluster ID to the last column of the centroid
            centroid = np.append(centroid, cluster_id)
            
            # Append centroid to the centroids list
            centroids.append(centroid)

        # Create a new array to store the extended point cloud with centroid information
        extended_point_cloud = np.zeros((point_cloud.shape[0], point_cloud.shape[1] + len(centroids[0]) - 1))
        extended_point_cloud[:, :-len(centroids[0]) + 1] = point_cloud  # Copy original point cloud data

        # Add centroid information to each row in the original point cloud
        for row_idx in range(point_cloud.shape[0]):
            cluster_id = point_cloud[row_idx, -1]
            centroid = next((c for c in centroids if c[-1] == cluster_id), None)
            
            if centroid is not None:
                extended_point_cloud[row_idx] = np.append(point_cloud[row_idx], centroid[:-1])  # Append centroid without the cluster ID
            else:
                extended_point_cloud[row_idx] = np.append(point_cloud[row_idx], np.zeros(len(centroids[0]) - 1))  # Fill with zeros if centroid is not found

        return extended_point_cloud
    

    
    def update_missing_semantic_ids(self, prev_cloud, curr_cloud):
        
        prev_coords = prev_cloud[:, :3]
        prev_semantic_ids = prev_cloud[:, 3]
        curr_coords = curr_cloud[:, :3]
        curr_semantic_ids = curr_cloud[:, 3]

        # Discretize the point cloud by cluster centroids
        # cluster_id, x, y, z
        unique_curr_cluster = np.unique(curr_cloud[:, -4:], axis=0)
        unique_prev_cluster = np.unique(prev_cloud[:, -4:], axis=0)

        # Find the closest cluster in the previous cloud for each cluster in the current cloud
        distances = cdist(unique_curr_cluster[:, -3:], unique_prev_cluster[:, -3:])
        closest_indices = np.argmin(distances, axis=1)
        closest_prev_cluster_ids = unique_prev_cluster[closest_indices, 0]  # Cluster IDs are in the first column

        # Update semantic IDs for clusters in the current cloud that have invalid semantic IDs
        invalid_semantic_ids_indices = np.where((curr_cloud[:, 3] == -1) | (curr_cloud[:, 3] == 255))[0]

        for index in invalid_semantic_ids_indices:
            curr_cluster_centroid = curr_cloud[index, -3:]  # x_cluster_centroid, y_cluster_centroid, z_cluster_centroid
            closest_index = np.argmin(np.linalg.norm(unique_curr_cluster[:, -3:] - curr_cluster_centroid, axis=1))
            closest_prev_id = closest_prev_cluster_ids[closest_index]

            # Find the semantic ID associated with the closest cluster in the previous cloud
            matching_indices = np.where(prev_cloud[:, 4] == closest_prev_id)[0]
            if len(matching_indices) > 0:
                matched_semantic_id = np.bincount(prev_cloud[matching_indices, 3].astype(int)).argmax()
                curr_cloud[index, 3] = matched_semantic_id

        return curr_cloud
                    
        
        
    def update(self, image, pointcloud, calib):
 
        # x y z intensity Xx4
        curr_pc = pointcloud.copy()
        curr_pc = clip_pointcloud(curr_pc)  # todo .. clip pointcloud to a specific range from yaml
        
        # Model prediction segmentation
        processed_img = preprocessing_kitti(image)
        semantic = self.model(processed_img)
        semantic = postprocessing(semantic)

        # Assign semantic class to each point in the point cloud12
        # Res : x y z semantic_class
        segmented_pointcloud = self.paint.paint(curr_pc, semantic, calib)

        # set semantic_class 255 if x  is -ve
        segmented_pointcloud[segmented_pointcloud[:, 0] < 0, 3] = 255
        
        # Clustering
        # Res : x y z semantic_class cluster_id
        clustered = self.dbscan.fit_predict(segmented_pointcloud[:, :3])  # Considering only x, y, z coordinates for clustering
        clustered_pc = np.concatenate((segmented_pointcloud, clustered[:, None]), axis=1)
        
        # Append cluster centroids to the point cloud
        # Res : x y z semantic_class cluster_id centroid_x centroid_y centroid_z
        clustered_pc = self.append_cluster_centroids(clustered_pc)
        
        if self.prev_pcd is not None:
            clustered_pc = self.update_missing_semantic_ids(self.prev_pcd, clustered_pc, threshold=1)
        
        self.prev_pcd = clustered_pc.copy()
        
        self.segmented_img = self.semantic_to_color(semantic)
        self.map = self.lidar2map(clustered_pc)
    
    
    def get_map(self):
        """
            Return the map
        """
        return self.map
    
    def get_segmentation(self):
        """
            Return the segnmentation
        """
        return self.segmented_img