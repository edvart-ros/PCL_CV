import open3d as o3d

def get_img_simple(geometries):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False) #works for me with False, on some systems needs to be true
    for geometry in geometries:
        vis.add_geometry(geometry)
        vis.update_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer()
    vis.destroy_window()
    return img