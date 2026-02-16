import time
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
        self.waypoint_reward = 0.0 # NEW: Track impulse reward
        self._DISTANCE_TO_CENTERLINE_THRESHOLD = 3.0 # Meters before we consider it "off-road"
        
        # 2. Define Observation and Action Spaces
        self.observation_space = spaces.Dict({
            "depth": spaces.Box(low=0, high=255, shape=(160, 160, 1), dtype=np.uint8),
            "semantic": spaces.Box(low=0, high=255, shape=(160, 160, 1), dtype=np.uint8),
            "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "goal": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32) 
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 3. Internal State
        self.vehicle = None
        self.sensors = []
        self.last_data = {"depth": None, "semantic": None}
    
    def reset(self):
        self._cleanup()
        time.sleep(1.0)
        self.stuck_ticks = 0
        self.waypoint_reward = 0.0
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        # --- Spawn Vehicle ---
        bp = self.blueprint_library.find('vehicle.tesla.model3')
        self.map = self.world.get_map() 
        spawn_point = random.choice(self.map.get_spawn_points())
        self.vehicle = self.world.spawn_actor(bp, spawn_point)
        
        # --- Generate Route (THE FIX) ---
        # Do NOT ask self.vehicle.get_location() here. It returns (0,0,0).
        # Use spawn_point.location directly.
        current_w = self.map.get_waypoint(spawn_point.location)
        self.route_waypoints = [current_w]
        
        # Generate the rest of the route
        for _ in range(5000):
            # Safe check in case the map ends
            next_ws = self.route_waypoints[-1].next(2.0)
            if len(next_ws) > 0:
                self.route_waypoints.append(next_ws[0])
            else:
                break

        self.current_waypoint_index = 1 

        # 4. Attach Multi-Modal Sensors
        self._setup_sensors()
        
        # 5. Stability Ticks (Crucial for Physics)
        # We tick BEFORE asking for observations so the car actually falls to the ground
        for _ in range(20):
            self.world.tick()
        
        # 6. Initialize Data
        max_tries = 100
        tries = 0
        while (self.last_data["depth"] is None or self.last_data["semantic"] is None) and tries < max_tries:
            self.world.tick()
            tries += 1
            
        self.collision_hist = []
        return self._get_obs()

    def step(self, action):
        throttle_val = float((action[1] + 1) / 2) 
        brake_val = 0.0
        control = carla.VehicleControl(
            steer=float(action[0]),
            throttle=float(max(0, throttle_val)),
            brake=float(max(0, brake_val))
        )
        self.vehicle.apply_control(control)

        self.world.tick()
        
        # 1. Check Waypoint Logic FIRST (so reward is ready)
        self._check_waypoint_completion()

        obs = self._get_obs()
        reward = self._calculate_reward()
        done = self._check_done()
        
        if done:
            self.collision_hist = []
        
        return obs, reward, done, {}

    def _check_waypoint_completion(self):
        """
        New Logic: Only advance index if we are close to the target.
        """
        self.waypoint_reward = 0.0 # Reset every step
        
        if self.current_waypoint_index >= len(self.route_waypoints) - 1:
            return

        # Get current target location
        target_loc = self.route_waypoints[self.current_waypoint_index].transform.location
        vehicle_loc = self.vehicle.get_location()
        
        # Distance check
        dist = vehicle_loc.distance(target_loc)
        
        # If within 1.0 meters, we hit it!
        if dist < 1.0:
            self.current_waypoint_index += 1
            self.waypoint_reward = 1.0 # The "Cookie" for progress!
            # print(f"Waypoint {self.current_waypoint_index} Reached! (+1.0)")

    def _setup_sensors(self):
        depth_bp = self.blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '160')
        depth_bp.set_attribute('image_size_y', '160')
        
        sem_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        sem_bp.set_attribute('image_size_x', '160')
        sem_bp.set_attribute('image_size_y', '160')

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

    # def _process_depth(self, image):
    #     array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    #     array = np.reshape(array, (image.height, image.width, 4))
    #     self.last_data["depth"] = array[:, :, 0:1] 
    
    def _process_depth(self, image):
        # This turns 24-bit raw depth into a 0-255 grayscale log map
        image.convert(carla.ColorConverter.LogarithmicDepth) 
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        self.last_data["depth"] = array[:, :, 0:1]

    def _process_sem(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        self.last_data["semantic"] = array[:, :, 2:3] 

    def _get_obs(self):
        # NOTE: Waypoint index logic moved to _check_waypoint_completion
        
        player_transform = self.vehicle.get_transform()
        target_waypoint = self.route_waypoints[self.current_waypoint_index]
        local_goal = self.get_local_goal(player_transform, target_waypoint.transform.location)
        
        v = self.vehicle.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2 + v.z**2)
        vector = np.array([speed, 0, 0], dtype=np.float32) 

        return {
            "depth": self.last_data["depth"],
            "semantic": self.last_data["semantic"],
            "vector": vector,
            "goal": local_goal / 10.0 
        }
    
    def get_local_goal(self, player_transform, goal_location):
        dx_world = goal_location.x - player_transform.location.x
        dy_world = goal_location.y - player_transform.location.y
        yaw = np.radians(player_transform.rotation.yaw)
        dx_local =  dx_world * np.cos(yaw) + dy_world * np.sin(yaw)
        dy_local = -dx_world * np.sin(yaw) + dy_world * np.cos(yaw)
        return np.array([dx_local, dy_local], dtype=np.float32)
    
    def _get_dist_to_centerline(self):
        if self.current_waypoint_index < 1:
            return 0.0
            
        A_loc = self.route_waypoints[self.current_waypoint_index - 1].transform.location
        B_loc = self.route_waypoints[self.current_waypoint_index].transform.location
        P_loc = self.vehicle.get_location()

        # 1. Vector AP (Car to Start) and AB (Segment Direction)
        AP = np.array([P_loc.x - A_loc.x, P_loc.y - A_loc.y])
        AB = np.array([B_loc.x - A_loc.x, B_loc.y - A_loc.y])
        
        # 2. Standard CTE Formula using Cross Product
        # Area of parallelogram / Base = Height (Distance)
        # We use absolute value because distance is always positive
        cross_prod = np.abs(AP[0] * AB[1] - AP[1] * AB[0])
        norm_AB = np.linalg.norm(AB) + 1e-6
        
        return cross_prod / norm_AB
        
    def _calculate_reward(self):
        v = self.vehicle.get_velocity()
        speed_kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        # 1. CTE (Centerline Distance)
        cte = self._get_dist_to_centerline()
        
        # 2. SPEED (Targeting 25km/h)
        r_speed = np.sqrt(speed_kmh / 25.0) if speed_kmh < 25 else 1.0 

        # 3. CENTER (Penalty based on CTE)
        r_center = max(0.0, 1.0 - (cte / self._DISTANCE_TO_CENTERLINE_THRESHOLD))

        # 4. ANGLE (Alignment with the line segment)
        target_wp = self.route_waypoints[self.current_waypoint_index]
        v_yaw = self.vehicle.get_transform().rotation.yaw
        w_yaw = target_wp.transform.rotation.yaw
        diff = abs(v_yaw - w_yaw) % 360
        if diff > 180: diff = 360 - diff
        r_angle = max(0.0, 1.0 - (diff / 30.0)) 

        # Penalties
        r_collision = -15.0 if len(self.collision_hist) > 0 else 0.0
        r_stall = -10.0 if self.stuck_ticks >= 100 else 0.0 
        r_idle = -0.05 if speed_kmh < 1.0 else 0.0
        r_offroad = -10.0 if cte > self._DISTANCE_TO_CENTERLINE_THRESHOLD else 0.0 

        total_reward = (r_speed * r_center * r_angle) + r_collision + r_stall + r_offroad + r_idle + self.waypoint_reward
        return total_reward

    def _check_done(self):
        if self.current_waypoint_index >= len(self.route_waypoints) - 10:
            return True
        if len(self.collision_hist) > 0:
            return True
        
        v = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
        self.stuck_ticks = self.stuck_ticks + 1 if speed < 1.0 else 0

        if self.stuck_ticks > 100 or self._get_dist_to_centerline() > self._DISTANCE_TO_CENTERLINE_THRESHOLD:
            return True
            
        return False

    def _cleanup(self):
        for s in self.sensors:
            if s.is_alive:
                s.stop()
                s.destroy()
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.destroy()
        self.sensors = []
        self.vehicle = None
        time.sleep(0.5)