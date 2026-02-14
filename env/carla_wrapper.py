import carla
import numpy as np
import cv2
import random
import gymnasium as gym
from gymnasium import spaces

class CarlaEnv(gym.Env):
    def __init__(self, host='127.0.0.1', port=2000):
        super(CarlaEnv, self).__init__()
        # 1. Setup Client and World
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.stuck_ticks = 0
        
        # 2. Define Observation and Action Spaces
        # We use 160x160 as per DreamerV3 defaults for efficiency
        self.observation_space = spaces.Dict({
            "depth": spaces.Box(low=0, high=255, shape=(160, 160, 1), dtype=np.uint8),
            "semantic": spaces.Box(low=0, high=255, shape=(160, 160, 1), dtype=np.uint8),
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        })
        # Continuous Control: [Steer, Throttle/Brake]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 3. Internal State
        self.vehicle = None
        self.sensors = []
        self.last_data = {"depth": None, "semantic": None}

    def reset(self):
        # 1. Cleanup existing actors
        self._cleanup()
        self.stuck_ticks = 0
        
        # 2. Configure Synchronous Mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        # --- MOVED UP: Spawn Vehicle FIRST ---
        bp = self.blueprint_library.find('vehicle.tesla.model3')
        # Get map here so we can find spawn points
        self.map = self.world.get_map() 
        spawn_point = random.choice(self.map.get_spawn_points())
        self.vehicle = self.world.spawn_actor(bp, spawn_point)
        # -------------------------------------
        
        # --- NOW you can Generate the Route ---
        # Now self.vehicle exists, so this line won't crash
        current_w = self.map.get_waypoint(self.vehicle.get_location())
        
        self.route_waypoints = [current_w]
        for _ in range(5000):
            # next(2.0) returns a list; pick the first one
            next_w = self.route_waypoints[-1].next(2.0)[0]
            self.route_waypoints.append(next_w)
        
        self.current_waypoint_index = 0
        # -------------------------------------

        # 4. Attach Multi-Modal Sensors
        self._setup_sensors()
        
        # 5. Tick to initialize data
        max_tries = 100
        tries = 0
        
        while (self.last_data["depth"] is None or self.last_data["semantic"] is None) and tries < max_tries:
            self.world.tick()
            tries += 1
        
        if tries == max_tries:
            print("ERROR: Sensors failed to initialize after 100 ticks.")
        
        self.collision_hist = []
        return self._get_obs()

    def step(self, action):
        # Apply Actions
        control = carla.VehicleControl(
            steer=float(action[0]),
            throttle=float(max(0, action[1])),
            brake=float(max(0, -action[1]))
        )
        self.vehicle.apply_control(control)

        # Tick the World
        self.world.tick()
        
        obs = self._get_obs()
        reward = self._calculate_reward()
        done = self._check_done()
        
        if done:
            self.collision_hist = []
        
        return obs, reward, done, {}

    def _setup_sensors(self):
        # Depth Camera
        depth_bp = self.blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '160')
        depth_bp.set_attribute('image_size_y', '160')
        
        # Semantic Segmentation Camera
        sem_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        sem_bp.set_attribute('image_size_x', '160')
        sem_bp.set_attribute('image_size_y', '160')

        # Spawn and Listen
        transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.depth_sensor = self.world.spawn_actor(depth_bp, transform, attach_to=self.vehicle)
        self.sem_sensor = self.world.spawn_actor(sem_bp, transform, attach_to=self.vehicle)
        
        self.depth_sensor.listen(lambda image: self._process_depth(image))
        self.sem_sensor.listen(lambda image: self._process_sem(image))
        
        self.sensors.extend([self.depth_sensor, self.sem_sensor])
        
        col_bp = self.blueprint_library.find('sensor.other.collision')
        self.col_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.col_sensor.listen(lambda event: self._on_collision(event))
        self.sensors.append(self.col_sensor)
        self.collision_hist = []
    
    def _on_collision(self, event):
        self.collision_hist.append(event)

    def _process_depth(self, image):
        # Convert raw to log-depth for better feature learning
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        self.last_data["depth"] = array[:, :, 0:1] # Red channel contains depth

    def _process_sem(self, image):
        # Raw tags (0-22) for road, vehicles, etc.
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        self.last_data["semantic"] = array[:, :, 2:3] # Blue channel contains tags

    def _get_obs(self):
        player_transform = self.vehicle.get_transform()
        
        # 1. Update Waypoint Index
        if self.current_waypoint_index < len(self.route_waypoints) - 5:
            self.current_waypoint_index += 1
        
        # 2. Calculate Goal
        target_waypoint = self.route_waypoints[self.current_waypoint_index]
        local_goal = self.get_local_goal(player_transform, target_waypoint.transform.location)
        
        # 3. Calculate Vector (Speed/Heading info)
        v = self.vehicle.get_velocity()
        # Vector: [Velocity X, Velocity Y, Angular Velocity Z] (Example)
        # Or just use [Speed, 0, 0] if you want simple speed
        speed = np.sqrt(v.x**2 + v.y**2 + v.z**2)
        vector = np.array([speed, 0, 0], dtype=np.float32) # Simplified for now

        return {
            "depth": self.last_data["depth"],       # FIX: Use self.last_data
            "semantic": self.last_data["semantic"], # FIX: Use self.last_data
            "vector": vector,                       # FIX: Added missing vector
            "goal": local_goal / 10.0 
        }
    
    def get_local_goal(self, player_transform, goal_location):
        """
        Transforms a global goal_location into a local (dx, dy) 
        relative to the player_transform (car's position/rotation).
        """
        # 1. Get relative world position
        dx_world = goal_location.x - player_transform.location.x
        dy_world = goal_location.y - player_transform.location.y
        
        # 2. Get car's rotation in radians
        # CARLA uses degrees; 0 is East, 90 is South
        yaw = np.radians(player_transform.rotation.yaw)
        
        # 3. Rotate world coordinates into car's local frame
        # Standard 2D rotation: 
        # x_local =  dx*cos(yaw) + dy*sin(yaw)
        # y_local = -dx*sin(yaw) + dy*cos(yaw)
        dx_local =  dx_world * np.cos(yaw) + dy_world * np.sin(yaw)
        dy_local = -dx_world * np.sin(yaw) + dy_world * np.cos(yaw)
        
        return np.array([dx_local, dy_local], dtype=np.float32)
        
    def _calculate_reward(self):
        v = self.vehicle.get_velocity()
        speed_kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        # 1. SPEED REWARD (Target: 25 km/h)
        # We prefer speed up to 25, then diminishing returns.
        if speed_kmh < 25:
            r_speed = np.sqrt(speed_kmh / 25.0)
        else:
            r_speed = 1.0 # Cap it so it doesn't prioritize speeding over safety

        # 2. CENTERING REWARD
        # Get distance from lane center
        vehicle_location = self.vehicle.get_location()
        waypoint = self.map.get_waypoint(vehicle_location)
        
        # Calculate distance between car and waypoint
        dist_to_center = vehicle_location.distance(waypoint.transform.location)
        
        # Reward is 1.0 if perfectly centered, drops to 0.0 at 2 meters out
        r_center = max(0.0, 1.0 - (dist_to_center / 2.0))

        # 3. ANGLE REWARD (Alignment)
        # Compare car heading vs road heading
        vehicle_yaw = self.vehicle.get_transform().rotation.yaw
        waypoint_yaw = waypoint.transform.rotation.yaw
        
        # Normalize angle difference to [-180, 180]
        diff = abs(vehicle_yaw - waypoint_yaw) % 360
        if diff > 180: diff = 360 - diff
        
        # Reward 1.0 if aligned, 0.0 if 90 degrees off
        r_angle = max(0.0, 1.0 - (diff / 30.0)) # Strict: must be within 30 deg

        # 4. COLLISION PENALTY
        r_collision = 0.0
        if len(self.collision_hist) > 0:
            r_collision = -15.0 # Massive penalty for crashing
        
        # 5. STALL PENALTY
        r_stall = -10.0 if self.stuck_ticks >= 100 else 0.0 
        
        r_offroad = 0.0
        # We use the same 3.0m threshold as your _check_done
        if dist_to_center > 3.0:
            r_offroad = -10.0 # Significant punishment for leaving the drivable area

        # TOTAL REWARD
        # Weighted sum: Speed is good, but Centering/Angle are multipliers
        # If you are off-road (r_center=0), speed reward becomes irrelevant.
        total_reward = r_speed * r_center * r_angle + r_collision + r_stall + r_offroad
        
        return total_reward

    def _check_done(self):
        if self.current_waypoint_index >= len(self.route_waypoints) - 10:
            print("SUCCESS: Route Completed!")
            return True
    
        # 1. Collision
        if len(self.collision_hist) > 0:
            return True
        
        v = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
        if speed < 1.0: # If slower than 1 km/h
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0

        if self.stuck_ticks > 100: # Stuck for 5 seconds (100 ticks * 0.05s)
            return True
            
        # 2. Off-road (Lane Invasion)
        vehicle_location = self.vehicle.get_location()
        waypoint = self.map.get_waypoint(vehicle_location)
        dist_to_center = vehicle_location.distance(waypoint.transform.location)
        
        # If car is more than 3 meters from center, it's off the road
        if dist_to_center > 3.0:
            return True
            
        return False

    def _cleanup(self):
        # Ensure clean break to avoid crashes
        for s in self.sensors:
            s.stop()
            s.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        self.sensors = []
        self.vehicle = None

if __name__ == "__main__":
    # 1. Initialize environment
    env = CarlaEnv()
    try:
        print("Resetting environment...")
        obs = env.reset()
        
        print(f"Observation shapes:")
        print(f"- Depth: {obs['depth'].shape}") # Should be (160, 160, 1)
        print(f"- Semantic: {obs['semantic'].shape}") # Should be (160, 160, 1)
        
        for i in range(10):
            # 2. Take a step (Steer 0.1, Throttle 0.5)
            action = np.array([0.1, 0.5])
            obs, reward, done, _ = env.step(action)
            
            # 3. Visual Verification using OpenCV
            # Normalize for display
            depth_vis = cv2.applyColorMap(obs['depth'], cv2.COLORMAP_JET)
            # Multiply tags by 10 so they are visible (0-22 is too dark raw)
            sem_vis = cv2.applyColorMap(obs['semantic'] * 10, cv2.COLORMAP_JET)
            
            cv2.imshow("Depth Stream", depth_vis)
            cv2.imshow("Semantic Stream", sem_vis)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            print(f"Step {i}: Reward (Speed) = {reward:.2f} km/h")
            
    finally:
        # 4. Clean exit to prevent the "Not Responding" crash
        print("Cleaning up...")
        cv2.destroyAllWindows()
        env._cleanup()
        # Set back to async before closing script
        settings = env.world.get_settings()
        settings.synchronous_mode = False
        env.world.apply_settings(settings)