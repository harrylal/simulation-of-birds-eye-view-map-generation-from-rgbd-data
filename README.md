
<h1 align="center">
  Simulation of Birds Eye View Map Generation from RGB-D Data
  <br>
</h1>

<h4 align="center">This project aims to employ sensor fusion techniques utilizing data from lidar and camera sensors within the KITTI dataset to generate an informative birds-eye-view occupancy grid map enriched with semantic details</h4>
<br>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#download">Download</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
    <img src="assets/ou1.gif" alt="dem1">
    <br> 
    <br> 
    <br>
    <img src="assets/ou2.gif" alt="dem2">
    <br> 
    <br> 
    <br>
    <img src="assets/ou4.gif" alt="dem2">
    <br> 
    <br> 
    <br>
    <img src="assets/ou5.gif" alt="dem2">
</p>

<br>

## Key Features

1. **Bird's Eye View Occupancy Grid Map Generation:** Creation of a comprehensive occupancy grid map from LiDAR data providing a top-down view.
2. **Semantic Segmentation Mask Prediction using BiSeNetv2:** Employing BiSeNetv2 for predicting semantic segmentation masks of the identified objects.
  
3. **Point Cloud Projection onto Image for Semantic Information:** Projection of the point cloud overlapping the camera's field of view onto images to extract semantic information about objects.
  
4. **DBSCAN-Based Object Clustering in Point Cloud:** Implementation of DBSCAN for clustering objects within the point cloud data.
  
5. **Temporal Tracking of Objects for Semantic Identity Retention:** Tracking the identified objects in a temporal domain, ensuring semantic identity retention even if objects move out of the camera's field of view.


## Download
1. Download KITTI data files and caliberation files [KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php)
2. Download BiSeNet Weights [Drive](https://drive.google.com/file/d/10-WxqSmyFKW72_1D-2vwu7BzUFlCOwgb/view?usp=sharing)


## How To Use

1. Set the path to the downloaded KITTI dataset files in the `config/settings.yaml` file.
2. Set Bisenet Weights Path in `config/settings.yaml` file.
3. Execute the `main.py` file to start the project. 
    ```bash
    python3 main.py
    ```

## Credits
- [BiSeNet](https://arxiv.org/abs/2004.02147)
- [KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php)
- [PointPainting](https://github.com/AmrElsersy/PointPainting)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.





