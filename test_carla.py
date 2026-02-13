import carla
import random
import time

def main():
    vehicle = None
    client = None
    
    try:
        # 1. Connect to the client
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        # 2. Get the world and blueprint library
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        
        # 3. Enable Synchronous Mode (Critical for keeping logic and physics in step)
        settings = world.get_settings()
        settings.synchronous_mode = True 
        settings.fixed_delta_seconds = 0.05  # Sets a stable 20 FPS
        world.apply_settings(settings)
        
        # 4. Setup Traffic Manager in Synchronous Mode
        # This is mandatory if synchronous_mode is True and you use Autopilot
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        
        # 5. Find and Spawn a Tesla Model 3
        bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        vehicle = world.spawn_actor(bp, spawn_point)
        print(f"Successfully spawned {vehicle.type_id} at {spawn_point.location}")
        
        # 6. Enable Autopilot
        vehicle.set_autopilot(True)
        
        # 7. Get the Spectator for the Chase Cam
        spectator = world.get_spectator()
        
        # 8. The Simulation Loop
        print("Running simulation for 30 seconds (600 ticks)...")
        for _ in range(600):
            # Tell the server to calculate one frame
            world.tick() 
            
            # --- Chase Camera Logic ---
            v_transform = vehicle.get_transform()
            # Calculate position: 8 meters back, 4 meters up
            # We use the vehicle's forward vector to determine "backward"
            forward_vec = v_transform.get_forward_vector()
            back_pos = v_transform.location - (forward_vec * 8.0) + carla.Location(z=4.0)
            
            # Update the spectator camera to look at the car
            spectator.set_transform(carla.Transform(back_pos, v_transform.rotation))
            
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        # 9. Graceful Cleanup
        if vehicle is not None:
            print("Cleaning up vehicle...")
            vehicle.destroy()
        
        # 10. CRITICAL: Reset to Asynchronous mode
        # If this isn't done, the CARLA Editor/Server will hang waiting for a tick.
        if 'world' in locals():
            # One last tick to finalize destruction
            world.tick() 
            
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            
            # Also reset TM (best practice)
            if 'tm' in locals():
                tm.set_synchronous_mode(False)
                
            print("Simulator reset to Asynchronous mode. Safe to exit.")

if __name__ == '__main__':
    main()