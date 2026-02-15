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
        
        # --- Generate Route ---
        current_w = self.map.get_waypoint(self.vehicle.get_location())
        self.route_waypoints = [current_w]
        for _ in range(5000):
            next_w = self.route_waypoints[-1].next(2.0)[0]
            self.route_waypoints.append(next_w)
        
        self.current_waypoint_index = 1 # Start aiming for the *next* one

        # 4. Attach Multi-Modal Sensors
        self._setup_sensors()
        
        # 5. Tick to initialize data
        max_tries = 100
        tries = 0
        while (self.last_data["depth"] is None or self.last_data["semantic"] is None) and tries < max_tries:
            self.world.tick()
            tries += 1
        
        self.collision_hist = []
        return self._get_obs()

    def step(self, action):
        control = carla.VehicleControl(
            steer=float(action[0]),
            throttle=float(max(0, action[1])),
            brake=float(max(0, -action[1]))
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
        
        # If within 1.5 meters (generous radius), we hit it!
        if dist < 1.5:
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

    def _process_depth(self, image):
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
        
    def _calculate_reward(self):
        v = self.vehicle.get_velocity()
        speed_kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        # 1. SPEED
        if speed_kmh < 25:
            r_speed = np.sqrt(speed_kmh / 25.0)
        else:
            r_speed = 1.0 

        # 2. CENTER
        vehicle_location = self.vehicle.get_location()
        waypoint = self.map.get_waypoint(vehicle_location)
        dist_to_center = vehicle_location.distance(waypoint.transform.location)
        r_center = max(0.0, 1.0 - (dist_to_center / 2.0))

        # 3. ANGLE
        vehicle_yaw = self.vehicle.get_transform().rotation.yaw
        waypoint_yaw = waypoint.transform.rotation.yaw
        diff = abs(vehicle_yaw - waypoint_yaw) % 360
        if diff > 180: diff = 360 - diff
        r_angle = max(0.0, 1.0 - (diff / 30.0)) 

        # 4. COLLISION
        r_collision = 0.0
        if len(self.collision_hist) > 0:
            r_collision = -15.0 
        
        # 5. STALL
        r_stall = -10.0 if self.stuck_ticks >= 100 else 0.0 
        
        # 6. IDLE PENALTY
        r_idle = -0.05 if speed_kmh < 1.0 else 0.0

        # 7. OFFROAD
        r_offroad = 0.0
        if dist_to_center > 3.0:
            r_offroad = -10.0 

        # 8. WAYPOINT BONUS [NEW]
        # This comes from _check_waypoint_completion()
        r_waypoint = self.waypoint_reward

        # TOTAL
        total_reward = (r_speed * r_center * r_angle) + r_collision + r_stall + r_offroad + r_idle + r_waypoint
        
        return total_reward

    def _check_done(self):
        if self.current_waypoint_index >= len(self.route_waypoints) - 10:
            return True
    
        if len(self.collision_hist) > 0:
            return True
        
        v = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        if speed < 1.0:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0

        if self.stuck_ticks > 100:
            return True
            
        vehicle_location = self.vehicle.get_location()
        waypoint = self.map.get_waypoint(vehicle_location)
        dist_to_center = vehicle_location.distance(waypoint.transform.location)
        
        if dist_to_center > 3.0:
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