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
        # Cleanup existing actors
        self._cleanup()
        
        # Configure Synchronous Mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        # Spawn Vehicle
        bp = self.blueprint_library.find('vehicle.tesla.model3')
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(bp, spawn_point)

        # Attach Multi-Modal Sensors
        self._setup_sensors()
        
        # Tick to initialize data
        self.world.tick()
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
        v = self.vehicle.get_velocity()
        kmh = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
        return {
            "depth": self.last_data["depth"],
            "semantic": self.last_data["semantic"],
            "vector": np.array([kmh, 0.0, 0.0], dtype=np.float32) # Simplification
        }

    def _cleanup(self):
        # Ensure clean break to avoid crashes
        for s in self.sensors:
            s.stop()
            s.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        self.sensors = []
        self.vehicle = None