import time
import pygame
import numpy as np
from pygame.locals import RESIZABLE, VIDEORESIZE

# Add the CARLA Python API to PYTHONPATH
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


def setup_dual_view(initial_width=800, initial_height=1200):
    pygame.init()
    
    display = pygame.display.set_mode((initial_width, initial_height), RESIZABLE)
    pygame.display.set_caption("Vehicle Views (Resizable)")
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    vehicle = None
    while vehicle is None:
        vehicles = world.get_actors().filter('vehicle.*')
        for v in vehicles:
            if v.attributes.get('role_name') == 'hero':
                vehicle = v
                break
        time.sleep(0.1)

    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    
    def update_camera_resolution(width, height):
        nonlocal camera_bp
        camera_bp.set_attribute('image_size_x', str(width))
        camera_bp.set_attribute('image_size_y', str(height//2))
        return camera_bp
    
    camera_bp = update_camera_resolution(initial_width, initial_height)
    
    fpv_transform = carla.Transform(carla.Location(x=0.8, z=1.7))
    ghost_transform = carla.Transform(carla.Location(x=-15, z=25, y=0), 
                                    carla.Rotation(pitch=-65))
    
    fpv_camera = world.spawn_actor(camera_bp, fpv_transform, attach_to=vehicle)
    ghost_camera = world.spawn_actor(camera_bp, ghost_transform, attach_to=vehicle)
    
    image_fpv = None
    image_ghost = None
    
    def process_fpv_img(image):
        nonlocal image_fpv
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        image_fpv = array
        
    def process_ghost_img(image):
        nonlocal image_ghost
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        image_ghost = array
    
    fpv_camera.listen(process_fpv_img)
    ghost_camera.listen(process_ghost_img)
    
    current_width = initial_width
    current_height = initial_height
    
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == VIDEORESIZE:
                    current_width = event.w
                    current_height = event.h
                    display = pygame.display.set_mode((current_width, current_height), RESIZABLE)
                    
                    # Update cameras with new resolution
                    fpv_camera.destroy()
                    ghost_camera.destroy()
                    camera_bp = update_camera_resolution(current_width, current_height)
                    fpv_camera = world.spawn_actor(camera_bp, fpv_transform, attach_to=vehicle)
                    ghost_camera = world.spawn_actor(camera_bp, ghost_transform, attach_to=vehicle)
                    fpv_camera.listen(process_fpv_img)
                    ghost_camera.listen(process_ghost_img)
            
            if image_fpv is not None:
                surface_fpv = pygame.surfarray.make_surface(image_fpv.swapaxes(0, 1))
                scaled_fpv = pygame.transform.scale(surface_fpv, (current_width, current_height//2))
                display.blit(scaled_fpv, (0, 0))
                
            if image_ghost is not None:
                surface_ghost = pygame.surfarray.make_surface(image_ghost.swapaxes(0, 1))
                scaled_ghost = pygame.transform.scale(surface_ghost, (current_width, current_height//2))
                display.blit(scaled_ghost, (0, current_height//2))
            
            pygame.display.flip()
            
    finally:
        pygame.quit()
        fpv_camera.destroy()
        ghost_camera.destroy()

if __name__ == '__main__':
    try:
        setup_dual_view()
    except KeyboardInterrupt:
        pass
