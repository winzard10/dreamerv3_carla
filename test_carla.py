import carla
import random
import time

def main():
    vehicle = None
    try:
        # 1. Connect to the client
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        # 2. Get the world and blueprint library
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        
        # 3. Enable Synchronous Mode
        # This prevents CARLA from "running away" while your script is busy
        settings = world.get_settings()
        settings.synchronous_mode = True 
        settings.fixed_delta_seconds = 0.05 # Sets a stable 20 FPS
        world.apply_settings(settings)
        
        # 4. Find and Spawn a Tesla Model 3
        bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        vehicle = world.spawn_actor(bp, spawn_point)
        print(f"Successfully spawned {vehicle.type_id} at {spawn_point.location}")
        
        # 5. Enable Autopilot
        vehicle.set_autopilot(True)
        
        # 6. The Simulation Loop (Replacing time.sleep)
        # Instead of sleeping, we manually tell the server to calculate 200 frames
        # At 0.05s per frame, this is exactly 10 seconds of simulation time.
        print("Running simulation for 10 seconds...")
        for _ in range(200):
            world.tick() 
            
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        # 7. Graceful Cleanup
        if vehicle is not None:
            print("Cleaning up vehicle...")
            vehicle.destroy()
        
        # 8. CRITICAL: Tick one last time to process the destruction
        # and then disable synchronous mode so the simulator doesn't freeze.
        if 'world' in locals():
            world.tick() 
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            print("Simulator reset to Asynchronous mode. Safe to exit.")

if __name__ == '__main__':
    main()