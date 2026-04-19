import time
import carla
from cv2.gapi import threshold
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

# How many waypoints ahead to use as the goal target.
# Used for the dx/dy goal vector — points car toward near future.
from params import COLLECT_LOOKAHEAD as GOAL_LOOKAHEAD


class CarlaEnv(gym.Env):
    def __init__(self, town=None, host='127.0.0.1', port=2000):
        super(CarlaEnv, self).__init__()
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)

        if town:
            current_map = self.client.get_world().get_map().name
            if town not in current_map:
                print(f"Loading {town}...")
                self.client.load_world(town)

        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.stuck_ticks = 0
        self.waypoint_reward = 0.0
        self._DISTANCE_TO_CENTERLINE_THRESHOLD = 3.0
        self.lane_invasion_count = 0
        self.prev_action = None

        self.observation_space = spaces.Dict({
            "depth":    spaces.Box(low=0, high=255, shape=(128, 128, 1), dtype=np.uint8),
            "semantic": spaces.Box(low=0, high=255, shape=(128, 128, 1), dtype=np.uint8),
            "vector":   spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "goal":     spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.vehicle = None
        self.sensors = []
        self.last_data = {"depth": None, "semantic": None}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._cleanup()
        self.last_data = {"depth": None, "semantic": None}
        time.sleep(1.0)
        self.stuck_ticks = 0
        self.waypoint_reward = 0.0
        self.prev_action = None
        self.last_reward_components = {}
        self.lane_invasion_count = 0

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        bp = self.blueprint_library.find('vehicle.tesla.model3')
        self.map = self.world.get_map()
        spawn_point = random.choice(self.map.get_spawn_points())
        self.vehicle = self.world.spawn_actor(bp, spawn_point)

        current_w = self.map.get_waypoint(spawn_point.location)
        self.route_waypoints = [current_w]

        for _ in range(5000):
            next_ws = self.route_waypoints[-1].next(2.0)
            if len(next_ws) > 0:
                self.route_waypoints.append(next_ws[0])
            else:
                break

        self.current_waypoint_index = 1

        self._setup_sensors()

        for _ in range(20):
            self.world.tick()

        max_tries = 100
        tries = 0
        while (self.last_data["depth"] is None or self.last_data["semantic"] is None) and tries < max_tries:
            self.world.tick()
            tries += 1

        self.collision_hist = []
        return self._get_obs(), {}

    def step(self, action):
        throttle_val = float((action[1] + 1) / 2)
        control = carla.VehicleControl(
            steer=float(action[0]),
            throttle=float(max(0, throttle_val)),
            brake=0.0,
        )
        self.vehicle.apply_control(control)
        self.world.tick()

        self._check_waypoint_completion()

        obs = self._get_obs()
        reward = self._calculate_reward(action)
        terminated = self._check_done()
        truncated = False
        done = terminated or truncated

        if done:
            self.collision_hist = []

        info = {
            "reward_components": getattr(self, "last_reward_components", {}).copy(),
            "lane_invasions": self.lane_invasion_count
        }
        return obs, float(reward), terminated, truncated, info

    # John's idea: use longitudinal projection to update waypoints, with two conditions:
    def _check_waypoint_completion(self):
        """
        Waypoint update with longitudinal projection.

        Two conditions to advance the waypoint index:
          1. Car enters radius R (1.5m) around the waypoint → advance + give reward
          2. Car passes the waypoint longitudinally (even if it missed the radius)
             → advance silently, no reward

        Condition 2 prevents the car from getting stuck chasing a waypoint
        it has already driven past, which would cause incorrect heading rewards
        and a stale goal vector.

        Diagram:
          prev_wp ----[target_wp]---- next_wp
                           |
                      blue line (perpendicular to route)
                      green circle (radius R=1.5m)

          If car crosses blue line → advance (no reward)
          If car enters green circle → advance + reward
          R > crossing threshold so reward requires actually being close
        """
        self.waypoint_reward = 0.0

        if self.current_waypoint_index >= len(self.route_waypoints) - 1:
            return

        target_loc  = self.route_waypoints[self.current_waypoint_index].transform.location
        vehicle_loc = self.vehicle.get_location()

        # Vector from target waypoint to vehicle
        t2v = vehicle_loc - target_loc

        # --- Condition 1: Car is within reward radius ---
        if t2v.length() < 1.5:
            self.current_waypoint_index += 1
            self.waypoint_reward = 1.0
            return

        # --- Condition 2: Car has passed the waypoint longitudinally ---
        # Only check if there is a previous waypoint to define the route direction
        if self.current_waypoint_index > 0:
            prev_loc = self.route_waypoints[self.current_waypoint_index - 1].transform.location

            # Vector from target waypoint back toward previous waypoint
            # This defines the "incoming route direction"
            t2p = prev_loc - target_loc
            t2p_len = t2p.length() + 1e-6  # avoid division by zero

            # Project t2v onto t2p (longitudinal component)
            # Positive scl means vehicle is on the far side of target
            # (i.e. has passed it in the route direction)
            scl = (t2p.x * t2v.x + t2p.y * t2v.y + t2p.z * t2v.z) / t2p_len

            # scl < 1.0 means vehicle is up to 1m past the waypoint longitudinally
            # Threshold of 1.0m is robust at typical driving speeds (25 km/h → 0.35m/tick)
            if scl < 1.0:
                self.current_waypoint_index += 1
                # No reward — car missed the radius, just advance to keep goal consistent

    def _get_obs(self):
        player_transform = self.vehicle.get_transform()

        # Goal: dx/dy to a waypoint GOAL_LOOKAHEAD steps ahead.
        # This is close enough (6m) to follow road curvature correctly,
        # while still giving the actor a forward-looking direction signal.
        # The car doesn't need to explicitly reach this waypoint —
        # it gets reward by passing through intermediate 2m waypoints.
        goal_idx = min(
            self.current_waypoint_index + GOAL_LOOKAHEAD,
            len(self.route_waypoints) - 1
        )
        target_waypoint = self.route_waypoints[goal_idx]
        local_goal = self.get_local_goal(
            player_transform,
            target_waypoint.transform.location
        )

        v = self.vehicle.get_velocity()
        speed_ms = np.sqrt(v.x**2 + v.y**2 + v.z**2)

        return {
            "depth":    self.last_data["depth"],
            "semantic": self.last_data["semantic"],
            # Normalized: speed in [0, ~1] for typical driving speeds
            "vector":   np.array([speed_ms / 10.0, 0, 0], dtype=np.float32),
            # Normalized: raw dx/dy in meters / 10.0
            # With GOAL_LOOKAHEAD=3 and 2m spacing, raw values are ~0-6m
            # After /10.0 they are ~0-0.6 — well scaled for the network
            "goal":     (local_goal / 10.0).astype(np.float32),
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

        AP = np.array([P_loc.x - A_loc.x, P_loc.y - A_loc.y])
        AB = np.array([B_loc.x - A_loc.x, B_loc.y - A_loc.y])

        cross_prod = np.abs(AP[0] * AB[1] - AP[1] * AB[0])
        norm_AB = np.linalg.norm(AB) + 1e-6

        return cross_prod / norm_AB

    def _calculate_reward(self, action):
        v = self.vehicle.get_velocity()
        speed_kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
        cte = self._get_dist_to_centerline()

        # ----------------------------------------------------------------
        # 1. PROGRESS — primary signal, +5 per waypoint reached
        # ----------------------------------------------------------------
        r_progress = 3.0 if self.waypoint_reward > 0 else 0.0

        # ----------------------------------------------------------------
        # 2. SPEED — linear ramp to target, gentle penalty above
        # ----------------------------------------------------------------
        target_speed = 15.0
        if speed_kmh < 1.0:
            r_speed = -0.5
        elif speed_kmh <= target_speed:
            r_speed = speed_kmh / target_speed
        else:
            r_speed = 1.0 - abs(speed_kmh - target_speed) / 5.0
            r_speed = max(-1.0, r_speed)

        # ----------------------------------------------------------------
        # 3. CENTERLINE — quadratic falloff, smooth near center
        # ----------------------------------------------------------------
        threshold = self._get_centerline_threshold()
        r_center = max(0.0, 1.0 - (cte / threshold) ** 2)

        # ----------------------------------------------------------------
        # 4. HEADING — alignment with current waypoint road direction
        # ----------------------------------------------------------------
        if speed_kmh > 2.0:
            target_wp = self.route_waypoints[self.current_waypoint_index]
            v_yaw = self.vehicle.get_transform().rotation.yaw
            w_yaw = target_wp.transform.rotation.yaw
            diff = abs(v_yaw - w_yaw) % 360
            if diff > 180:
                diff = 360 - diff
            r_heading = max(0.0, 1.0 - (diff / 45.0))
        else:
            r_heading = 0.0

        # ----------------------------------------------------------------
        # 5. CONTINUOUS DRIVING — additive, gated by speed
        #    r_center appears here only once (removed stability_gate that
        #    double-penalized centerline deviation).
        # ----------------------------------------------------------------
        r_center_eff = r_center * r_heading

        r_driving = (
            0.4 * r_center_eff
            + 0.4 * r_heading
        ) * min(1.0, speed_kmh / 5.0) + 0.1 * r_speed

        # ----------------------------------------------------------------
        # 6. CONTROL REGULARIZATION
        #    (a) Steering magnitude — mild penalty, doesn't prevent sharp turns
        #    (b) Control rate — penalize jerky changes (main smoothness signal)
        # ----------------------------------------------------------------
        steer        = float(action[0])
        throttle_cmd = float((action[1] + 1) / 2)

        # (a) Steering magnitude — gentle, allows sharp turns when needed
        r_ctrl_mag = (-0.002 * (steer ** 2) 
                    - 0.08 * (throttle_cmd ** 2)) 

        # (b) Control rate — main smoothness penalty
        if self.prev_action is not None:
            prev_steer        = float(self.prev_action[0])
            prev_throttle_cmd = float((self.prev_action[1] + 1) / 2)

            delta_steer    = steer - prev_steer
            delta_throttle = throttle_cmd - prev_throttle_cmd

            r_ctrl_rate = (
                -0.05 * (delta_steer ** 2) 
                -0.30 * (delta_throttle ** 2) 
            )
        else:
            r_ctrl_rate = 0.0

        # ----------------------------------------------------------------
        # 7. HARD PENALTIES
        # ----------------------------------------------------------------
        r_collision = -20.0 if len(self.collision_hist) > 0 else 0.0
        r_offroad   = -5.0  if cte > self._DISTANCE_TO_CENTERLINE_THRESHOLD else 0.0
        r_stall     = -5.0  if self.stuck_ticks >= 100 else 0.0

        self.prev_action = np.array(action, dtype=np.float32)
        
        self.last_reward_components = {
            "r_progress": float(r_progress),
            "r_speed": float(r_speed),
            "r_center": float(r_center),
            "r_heading": float(r_heading),
            "r_driving": float(r_driving),
            "r_ctrl_mag": float(r_ctrl_mag),
            "r_ctrl_rate": float(r_ctrl_rate),
            "r_collision": float(r_collision),
            "r_offroad": float(r_offroad),
            "r_stall": float(r_stall),
        }

        return float(
            r_progress
            + r_driving
            + r_ctrl_mag
            + r_ctrl_rate
            + r_collision
            + r_offroad
            + r_stall
        )

    def _check_done(self):
        if self.current_waypoint_index >= len(self.route_waypoints) - 10:
            return True
        if len(self.collision_hist) > 0:
            return True

        v = self.vehicle.get_velocity()
        speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
        self.stuck_ticks = self.stuck_ticks + 1 if speed < 1.0 else 0

        threshold = self._get_centerline_threshold()
        if self.stuck_ticks > 100 or self._get_dist_to_centerline() > threshold:
            return True

        return False

    def _setup_sensors(self):
        depth_bp = self.blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '128')
        depth_bp.set_attribute('image_size_y', '128')

        sem_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        sem_bp.set_attribute('image_size_x', '128')
        sem_bp.set_attribute('image_size_y', '128')

        transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.depth_sensor = self.world.spawn_actor(depth_bp, transform, attach_to=self.vehicle)
        self.sem_sensor   = self.world.spawn_actor(sem_bp,   transform, attach_to=self.vehicle)

        self.depth_sensor.listen(lambda image: self._process_depth(image))
        self.sem_sensor.listen(lambda image: self._process_sem(image))

        self.sensors.extend([self.depth_sensor, self.sem_sensor])

        col_bp = self.blueprint_library.find('sensor.other.collision')
        self.col_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.col_sensor.listen(lambda event: self._on_collision(event))
        self.sensors.append(self.col_sensor)
        self.collision_hist = []
        
        lane_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        self.lane_sensor = self.world.spawn_actor(
            lane_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.lane_sensor.listen(lambda event: self._on_lane_invasion(event))
        self.sensors.append(self.lane_sensor)

    def _on_lane_invasion(self, event):
        self.lane_invasion_count += 1

    def _on_collision(self, event):
        self.collision_hist.append(event)
        
    def _get_centerline_threshold(self):
        """
        Dynamic threshold based on lane width.

        Uses 10% margin on half lane width.
        Fallback to 1.1 * 1.8 if invalid.
        """
        try:
            if self.current_waypoint_index < len(self.route_waypoints):
                wp = self.route_waypoints[self.current_waypoint_index]
            else:
                wp = self.route_waypoints[-1]

            lane_width = float(wp.lane_width)

            if not np.isfinite(lane_width) or lane_width <= 0:
                raise ValueError

            half_width = lane_width / 2.0
            return 1.1 * half_width

        except:
            return 1.1 * 1.8  # fallback

    def _process_depth(self, image):
        image.convert(carla.ColorConverter.LogarithmicDepth)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        self.last_data["depth"] = array[:, :, 0:1]

    def _process_sem(self, image):
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        self.last_data["semantic"] = array[:, :, 2:3]

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