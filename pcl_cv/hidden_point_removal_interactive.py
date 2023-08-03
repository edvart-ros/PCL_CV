from utils import get_img_simple
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2


def custom_key_action(pcd_original):
    k = 0
    varying = False
    vis = o3d.visualization.VisualizerWithKeyCallback()
    pcd = copy.deepcopy(pcd_original)
    
    def space_callback(cis, action, mods):
        nonlocal k, varying
        if action == 1: # key down
            varying = True
            k += 1
        elif action == 0: # key up
            varying = False
            pass
        elif action == 2: # key repeat
            k += 1
        print(k)
        return True

    def backspace_callback(cis, action, mods):
        nonlocal k, varying
        if action == 1:
            k = 0
            varying = True
            print(k)
        elif action == 0:
            varying = False
        return True

    def animation_callback(vis):
        nonlocal k, varying, pcd
        if varying:
            if k == 0:
                pcd = pcd_original
            else:    
                _, pt_map = pcd_original.hidden_point_removal(camera, diameter*k)
                pcd = pcd_original.select_by_index(pt_map)
            vis.clear_geometries()
            vis.add_geometry(pcd)
            #varying = False
        return
    
    # key_action_callback will be triggered when theres a keyboard press
    vis.register_key_action_callback(32, space_callback)
    vis.register_key_action_callback(48, backspace_callback)
    # animation callback is always repeadetly called by the visualier
    vis.register_animation_callback(animation_callback)
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()


if __name__ == "__main__":
    path = o3d.data.BunnyMesh().path
    pcd = o3d.io.read_point_cloud(path)
    diameter = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    camera = [0, 0, diameter]

    custom_key_action(pcd)