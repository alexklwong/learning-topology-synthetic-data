#!bin/bash

mkdir -p data
mkdir -p data/virtual_kitti

wget http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar -P data/virtual_kitti
wget http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_depth.tar -P data/virtual_kitti

mkdir data/virtual_kitti/vkitti_2.0.3
tar -xvf data/virtual_kitti/vkitti_2.0.3_rgb.tar -C data/virtual_kitti/vkitti_2.0.3
tar -xvf data/virtual_kitti/vkitti_2.0.3_depth.tar -C data/virtual_kitti/vkitti_2.0.3
