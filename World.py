import pygame
import carla
from Utils.synch_mode import CarlaSyncMode
import Controller.PIDController as PIDController
import time
from Utils.utils import *
import math
import gym
import gymnasium as gym
from gymnasium import spaces
from Utils.HUD import HUD as HUD
from Utils.CubicSpline.cubic_spline_planner import *
import csv
import random


class World(gym.Env):
    def __init__(self, client, carla_world, hud, args, visuals=False):
        self.world = carla_world
        self.client = client
        self.map = self.world.get_map()
        self.hud = hud
        self.args = args
        self.waypoint_resolution = args.waypoint_resolution
        self.waypoint_lookahead_distance = args.waypoint_lookahead_distance
        self.desired_speed = args.desired_speed
        self.control_mode = args.control_mode
        self.controller = None
        self.control_count = 0.0
        self.random_spawn = 0
        self.world.on_tick(hud.on_world_tick)
        self.im_width = 640
        self.im_height = 480
        self.episode_start = 0
        self.visuals = visuals
        self.episode_reward = 0
        self.player = None
        self.parked_vehicle = None
        self.moving_vehicle1 = None
        self.moving_vehicle2 = None
        self.walker = None
        self.collision_sensor = None
        self.camera_rgb = None
        self.camera_rgb2 = None
        self.camera_rgb3 = None
        self.camera_rgb4 = None
        self.lane_invasion = None
        self._autopilot_enabled = False
        self._control = carla.VehicleControl()
        self.max_dist = 4.5
        self.counter = 0
        self.frame = None
        self.delta_seconds = 1.0 / args.FPS
        self.last_v = 0
        self.last_y = 0
        self.distance_parked = 100
        self.ttc_trigger = 1.0
        self.episode_counter = 0
        self.steer = 0
        self.last_steer = 0
        self.save_list = []
        # self.file_name = 'F:/E2E-CARLA-ReinforcementLearning-PPO/logs/1709073714-working-50kmh/evaluation/logger.csv'
        self.logger = False

        ## RL STABLE BASELINES
        self.action_space = spaces.Box(low=-1, high=1,shape=(2,),dtype="float")
        self.observation_space = spaces.Box(low=-0, high=255, shape=(128, 128, 4), dtype=np.uint8)


        self.global_t = 0 # global timestep


    def append_to_csv(self,file_name, data):
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)


    def reset(self, seed=None):

        self.destroy()

        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=1/self.args.FPS))
        self.episode_reward = 0
        self.desired_speed = self.args.desired_speed

        self.episode_counter += 1

        if self.logger:
            self.append_to_csv(file_name=self.file_name, data=self.save_list)
        self.save_list = []


        # CREATING ACTORS

        self.create_actors()


        velocity_vec = self.player.get_velocity()
        current_transform = self.player.get_transform()
        current_location = current_transform.location
        current_roration = current_transform.rotation
        current_x = current_location.x
        current_y = current_location.y
        current_yaw = wrap_angle(current_roration.yaw)
        current_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
        frame, current_timestamp = self.hud.get_simulation_information()
        self.controller.update_values(current_x, current_y, current_yaw, current_speed, current_timestamp, frame)
        self.episode_start = time.time()


        self.world.tick()             
        self.clock = pygame.time.Clock()
            
        ttc = self.time_to_collison()

        while ttc > self.ttc_trigger: #player_position < parked_position - self.realease_position:
            

            self.clock.tick_busy_loop(self.args.FPS)

            if self.parse_events(clock=self.clock, action=None):
                 return
            
            velocity_vec_st = self.player.get_velocity()
            current_speed = math.sqrt(velocity_vec_st.x**2 + velocity_vec_st.y**2 + velocity_vec_st.z**2)

            ttc = self.time_to_collison()
   
            snapshot, image_rgb,image_rgb2,image_rgb3,image_rgb4, lane, collision = self.synch_mode.tick(timeout=10.0)

            self.get_observation()
            #if want Third person view
            self.update_spectator_camera()

            if image_rgb is not None:
                img = process_img2(self, image_rgb)
            if image_rgb2 is not None:
                img2 = process_img2(self, image_rgb2)
            if image_rgb3 is not None:
                img3 = process_img2(self, image_rgb3)
            if image_rgb4 is not None:
                img4 = process_img2(self, image_rgb4)

        stacked_img = np.concatenate([img, img2, img3,img4], axis=2)
        last_transform = self.player.get_transform()
        last_location = last_transform.location
        self.last_y = last_location.y
        self.last_v = current_speed
        print(current_speed)

        return stacked_img,{}

    def update_spectator_camera(self):
        spectator = self.world.get_spectator()
        player_transform = self.player.get_transform()
        # Define an offset: behind the vehicle and a bit above.
        offset = carla.Location(x=-5, z=2.5)
        # Get the new location by transforming the offset from local to world coordinates.
        new_location = player_transform.transform(offset)
        # Set the new rotation, for instance, a slight downward pitch.
        new_rotation = carla.Rotation(
            pitch=player_transform.rotation.pitch - 10,
            yaw=player_transform.rotation.yaw,
            roll=0
        )
        # Create a new Transform object with the new location and rotation.
        spectator_transform = carla.Transform(new_location, new_rotation)
        # Apply the transform to the spectator.
        spectator.set_transform(spectator_transform)

    def tick(self, clock):
        self.hud.tick(self, clock)
        

    def destroy(self):
        self.world.tick()
            
        actors = [
            self.player,
            self.collision_sensor,
            self.camera_rgb,
            self.camera_rgb2,
            self.camera_rgb3,
            self.camera_rgb4,
            self.lane_invasion,
            self.parked_vehicle,
            self.moving_vehicle1,
            self.moving_vehicle2,
            self.walker]        
                           
        for actor in actors:
            if actor is not None:
                try:
                    actor.destroy()
                    self.world.tick()
                except:
                    pass


    def step(self, action):
        self.reward = 0
        done = False
        cos_yaw_diff = 0
        dist = 0
        collision = 0
        lane = 0
        traveled = 0

        if action is not None:
            self.counter += 1
            self.global_t += 1

            self.clock.tick_busy_loop(self.args.FPS)

            if self.apply_vehicle_control(action):
                return

            snapshot, image_rgb,image_rgb2,image_rgb3,image_rgb4, lane, collision = self.synch_mode.tick(timeout=10.0)
            self.get_observation()
            #if want Third person view
            #self.update_spectator_camera()
            cos_yaw_diff, dist, collision, lane, traveled,current_speed,jitter= self.get_reward_comp(self.player, self.spawn_waypoint, collision, lane)
            
            
            self.reward = self.reward_value(cos_yaw_diff, dist, collision, lane, traveled, 
                                            current_speed, jitter)
            self.episode_reward += self.reward

            if image_rgb is not None:
                img = process_img2(self, image_rgb)
            if image_rgb2 is not None:
                img2 = process_img2(self, image_rgb2)
            if image_rgb3 is not None:
                img3 = process_img2(self, image_rgb3)
            if image_rgb4 is not None:
                img4 = process_img2(self, image_rgb4)
            stacked_img = np.concatenate([img, img2, img3,img4], axis=2)
                

            if dist > self.max_dist:
                done = True

            vehicle_location = self.player.get_location()
            y_vh = vehicle_location.y
            if y_vh > float(self.args.spawn_y) + self.distance_parked + 15:
                self.reward += 50
                print("episode ended by reaching goal position")
                done = True

            truncated = False

            if collision == 1:
                done = True
                print("Episode ended by collision")
                
            if lane == 1:
                done = True
                self.reward -= 50
                print("Episode ended by lane invasion")
        
            if dist > self.max_dist:
                done = True
                self.reward -= 50
                print(f"Episode ended with dist from waypoint: {dist}")

            if current_speed < 0.1:
                done = True

        return stacked_img, self.reward, done, truncated, {}
    



    def get_reward_comp(self, vehicle, waypoint, collision, lane):
        vehicle_location = vehicle.get_location()
        x_wp = waypoint.transform.location.x
        y_wp = waypoint.transform.location.y

        x_vh = vehicle_location.x
        y_vh = vehicle_location.y

        wp_array = np.array([x_wp])
        vh_array = np.array([x_vh])

        dist = abs(np.linalg.norm(wp_array - vh_array))

        vh_yaw = correct_yaw(vehicle.get_transform().rotation.yaw)
        wp_yaw = correct_yaw(waypoint.transform.rotation.yaw)
        cos_yaw_diff = np.cos((vh_yaw - wp_yaw)*np.pi/180.)

        collision = 0 if collision is None else 1

        if lane is not None:
            lane_types = set(x.type for x in lane.crossed_lane_markings)
            text = ['%r' % str(x).split()[-1] for x in lane_types]
            lane = 1 if text[0] == "'Solid'" else 0
        
        elif lane is None:
            lane=0

        # lane = 0 if lane is None else 1

        traveled = y_vh - self.last_y
        # print(traveled)
 
        # finish = 1 if y_vh > -40 else 0
        
        ######
        velocity = vehicle.get_velocity()
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        # Compute jitter: the change in steering since the last time step.
        current_control = vehicle.get_control()
        current_steer = current_control.steer
        jitter = abs(current_steer - self.last_steer)
        # Update last steering value for next step
        self.last_steer = current_steer

        # Return additional components for use in reward calculation:
        # cos_yaw_diff, distance, collision, lane, traveled, current_speed, jitter
        return cos_yaw_diff, dist, collision, lane, traveled, current_speed, jitter
    
    def reward_value(self, cos_yaw_diff, dist, collision, lane, traveled, 
                 current_speed, jitter,
                 lambda_1=1, lambda_2=1, lambda_3=100, lambda_4=5, 
                 lambda_5=0.5, lambda_6=10, lambda_7=1):
        
        #desired_speed_value
        desired_speed = 0
        # Penalty for being below desired speed
        speed_penalty = max(0, desired_speed - current_speed)
        
        reward = (lambda_1 * cos_yaw_diff) \
                - (lambda_2 * dist) \
                - (lambda_3 * collision) \
                - (lambda_4 * lane) \
                + (lambda_5 * traveled) \
                #- (lambda_6 * speed_penalty) \
                #- (lambda_7 * jitter)
                
        return reward
    


    def time_to_collison(self):

         # EGO information
        velocity_vec = self.player.get_velocity()
        current_transform = self.player.get_transform()
        current_location = current_transform.location
        current_x = current_location.x
        current_y = current_location.y
        current_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
       

        #Parked vehicle information
        parked_transform = self.parked_vehicle.get_transform()
        velocity_parked = self.parked_vehicle.get_velocity()
        parked_location = parked_transform.location
        parked_x = parked_location.x
        parked_y = parked_location.y
        parked_speed = math.sqrt(velocity_parked.x**2 + velocity_parked.y**2 + velocity_parked.z**2)

        dist = np.sqrt((parked_y-current_y)**2 + (current_x-parked_x)**2)
        rel_speed = current_speed - parked_speed

        ttc = dist/rel_speed

        return np.abs(ttc)


    def parse_events(self, action, clock):

        if not self._autopilot_enabled:
            # Control loop
            # get waypoints
            current_location = self.player.get_location()
            velocity_vec = self.player.get_velocity()
            current_transform = self.player.get_transform()
            current_location = current_transform.location
            current_rotation = current_transform.rotation
            current_x = current_location.x
            current_y = current_location.y
            current_yaw = wrap_angle(current_rotation.yaw)
            current_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
            # print(f"Control input : speed : {current_speed}, current position : {current_x}, {current_y}, yaw : {current_yaw}")
            frame, current_timestamp =self.hud.get_simulation_information()
            ready_to_go = self.controller.update_values(current_x, current_y, current_yaw, current_speed, current_timestamp, frame)
            
            if ready_to_go:
                if self.control_mode == "PID":
                    current_location = self.player.get_location()
                    current_waypoint = self.map.get_waypoint(current_location).next(self.waypoint_resolution)[0]
                    # print(current_waypoint.transform.location.x-current_x)
                    # print(current_waypoint.transform.location.y-current_y)            
                    waypoints = []
                    for i in range(int(self.waypoint_lookahead_distance / self.waypoint_resolution)):
                        waypoints.append([current_waypoint.transform.location.x, current_waypoint.transform.location.y, self.desired_speed])
                        current_waypoint = current_waypoint.next(self.waypoint_resolution)[0]


                # print(f'wp real: {waypoints}')
                if action is not None:
                    waypoints_RL = self.get_cubic_spline_path(action, current_x=current_x, current_y=current_y)
                    self.print_waypoints(waypoints_RL)
                    # print(waypoints_RL)
                    self.controller.update_waypoints(waypoints_RL)
                else:
                    self.print_waypoints(waypoints)
                    self.controller.update_waypoints(waypoints)  

                self.controller.update_controls()
                self._control.throttle, self._control.steer, self._control.brake = self.controller.get_commands()
                # print(self._control)
                self.player.apply_control(self._control)
                self.control_count += 1

    
    def apply_vehicle_control(self, action):

        self.steer = action[0]
        print(f'steer = {self.steer}')
        self.acceleration = action[1]
        print(f'acceleration = {self.acceleration}')

        self._control.steer = self.steer

        if self.acceleration < 0:
             self._control.brake = np.abs(self.acceleration)
             self._control.throttle = 0

        else:
            self._control.throttle = self.acceleration
            self._control.brake = 0

        print(self._control)    

        self.player.apply_control(self._control)
        self.control_count += 1


    def print_waypoints(self, waypoints):

        for z in waypoints:
            spawn_location_r = carla.Location()
            spawn_location_r.x = float(z[0])
            spawn_location_r.y = float(z[1])
            spawn_location_r.z = 1.0
            self.world.debug.draw_string(spawn_location_r, 'O', draw_shadow=False,
                                                color=carla.Color(r=255, g=0, b=0), life_time=0.1,
                                                persistent_lines=True)
            



    def create_actors(self):

        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprint = self.blueprint_library.filter('*vehicle*')
        self.walker_blueprint = self.blueprint_library.filter('*walker.*')

        # PLAYER
    
        spawn_location = carla.Location()
        spawn_location.x = float(self.args.spawn_x)
        spawn_location.y = float(self.args.spawn_y)
        self.spawn_waypoint = self.map.get_waypoint(spawn_location)
        spawn_transform = self.spawn_waypoint.transform
        spawn_transform.location.z = 1.0
        self.player = self.world.try_spawn_actor(self.vehicle_blueprint.filter('model3')[0], spawn_transform)
        self.world.tick()   
        print('vehicle spawned')

        # Turn on position lights
        current_lights = carla.VehicleLightState.NONE
        current_lights |= carla.VehicleLightState.Position
        self.player.set_light_state(carla.VehicleLightState.Position)

        # CAMERA RGB 1

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{640}")
        self.rgb_cam.set_attribute("image_size_y", f"{480}")
        self.rgb_cam.set_attribute("fov", f"110")
        self.camera_rgb = self.world.spawn_actor(
            self.rgb_cam,
            carla.Transform(carla.Location(x=2, z=1), carla.Rotation(0,0,0)),
            attach_to=self.player)
        self.world.tick()

        # CAMERA RGB 2

        self.rgb_cam2 = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam2.set_attribute("image_size_x", f"{640}")
        self.rgb_cam2.set_attribute("image_size_y", f"{480}")
        self.rgb_cam2.set_attribute("fov", f"120")
        self.camera_rgb2 = self.world.spawn_actor(
            self.rgb_cam2,
            carla.Transform(carla.Location(x=0, y =1, z=1), carla.Rotation(0,90,0)),
            attach_to=self.player)
        self.world.tick()
        # # CAMERA RGB 3

        self.rgb_cam3 = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam3.set_attribute("image_size_x", f"{640}")
        self.rgb_cam3.set_attribute("image_size_y", f"{480}")
        self.rgb_cam3.set_attribute("fov", f"120")
        self.camera_rgb3 = self.world.spawn_actor(
            self.rgb_cam3,
            carla.Transform(carla.Location(x=0,y = -1, z=1), carla.Rotation(0,-90,0)),
            attach_to=self.player)
        self.world.tick()

        # # CAMERA RGB 4

        self.rgb_cam4 = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam4.set_attribute("image_size_x", f"{640}")
        self.rgb_cam4.set_attribute("image_size_y", f"{480}")
        self.rgb_cam4.set_attribute("fov", f"110")
        self.camera_rgb4 = self.world.spawn_actor(
            self.rgb_cam4,
            carla.Transform(carla.Location(x=-3, z=1), carla.Rotation(0,180,0)),
            attach_to=self.player)
        self.world.tick()

        # LANE SENSOR

        self.lane_invasion = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.lane_invasion'), 
            carla.Transform(), 
            attach_to=self.player)
        self.world.tick()

        # COLLISION SENSOR

        self.collision_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.collision'),
            carla.Transform(),
            attach_to=self.player)
        self.world.tick()
        
        
        # # LIDAR SENSOR
        # self.lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        # # Optionally, set lidar attributes (adjust as needed)
        # self.lidar_bp.set_attribute("range", "50")
        # self.lidar_bp.set_attribute("rotation_frequency", "10")
        # self.lidar_bp.set_attribute("channels", "32")
        # self.lidar_bp.set_attribute("points_per_second", "56000")
        # # Define a transform for the lidar sensor relative to the vehicle
        # lidar_transform = carla.Transform(carla.Location(x=0, z=2), carla.Rotation(pitch=0, yaw=0, roll=0))
        # self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, lidar_transform, attach_to=self.player)
        # self.world.tick()
        # print('Lidar sensor spawned')
        
        # SYNCH MODE CONTEXT

        self.synch_mode = CarlaSyncMode(self.world, self.camera_rgb,self.camera_rgb2,self.camera_rgb3,self.camera_rgb4, self.lane_invasion, self.collision_sensor)
        
        # STATIONARY CAR
        
        parking_position = carla.Transform(self.player.get_transform().location + carla.Location(0.5, self.distance_parked, 0), 
                                carla.Rotation(0,90,0))
        parked_vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.*'))
        self.parked_vehicle = self.world.spawn_actor(parked_vehicle_bp, parking_position)
        self.world.tick()
        
        # MOVING CARS
        
        # Define lane options (lateral offsets)
        lanes = [3.7, 7.3, 10.7]

        # ---------- MOVING VEHICLE 1 ----------
        # Pick a random lane for vehicle 1.
        lane1 = random.choice(lanes)
        offset_y1 = self.distance_parked - random.randint(15, 25)
        mv1_spawn_location = self.player.get_transform().location + carla.Location(x=lane1, y=offset_y1, z=0)
        mv1_transform = carla.Transform(mv1_spawn_location, carla.Rotation(0, 90, 0))
        mv1_bp = random.choice(self.blueprint_library.filter('*vehicle*'))
        self.moving_vehicle1 = self.world.try_spawn_actor(mv1_bp, mv1_transform)
        if self.moving_vehicle1 is not None:
            self.moving_vehicle1.apply_control(carla.VehicleControl(throttle=random.uniform(0.7, 0.8)))
        else:
            print("Failed to spawn moving_vehicle1")
        self.world.tick()

        # ---------- MOVING VEHICLE 2 ----------
        # For vehicle 2, choose a different lane to avoid lateral collision.
        remaining_lanes = [l for l in lanes if l != lane1]
        lane2 = random.choice(remaining_lanes)

        # We'll retry several times if needed to find a free spawn position.
        min_y_distance = 10  # minimum desired separation in y between vehicle 1 and vehicle 2
        mv2_spawned = False
        attempts = 0
        while not mv2_spawned and attempts < 5:
            offset_y2 = self.distance_parked - random.randint(40, 50)
            mv2_spawn_location = self.player.get_transform().location + carla.Location(x=lane2, y=offset_y2, z=0)
            # If vehicle 1 exists, ensure enough separation along y:
            if self.moving_vehicle1 is not None:
                mv1_location = self.moving_vehicle1.get_transform().location
                if abs(mv1_location.y - mv2_spawn_location.y) < min_y_distance:
                    attempts += 1
                    continue  # try a different offset
            mv2_transform = carla.Transform(mv2_spawn_location, carla.Rotation(0, 90, 0))
            mv2_bp = random.choice(self.blueprint_library.filter('*vehicle*'))
            self.moving_vehicle2 = self.world.try_spawn_actor(mv2_bp, mv2_transform)
            if self.moving_vehicle2 is not None:
                self.moving_vehicle2.apply_control(carla.VehicleControl(throttle=random.uniform(0.5, 0.6)))
                mv2_spawned = True
            else:
                attempts += 1
                print(f"Attempt {attempts} failed for moving_vehicle2.")
        if not mv2_spawned:
            print("Could not spawn moving_vehicle2 without collision.")
        self.world.tick()

        # ---------- WALKER ----------
        # For the walker, choose a lateral offset that is clearly different from the vehicle lanes.
        walker_spawned = False
        walker_attempts = 0
        while not walker_spawned and walker_attempts < 10:
            # Here we choose an offset that is unlikely to conflict (for example, farther from the vehicles).
            walker_lane_offset = random.choice([0, 5])
            walker_offset_y = self.distance_parked - random.randint(10, 20)
            walker_spawn_location = self.player.get_transform().location + carla.Location(x=walker_lane_offset, y=walker_offset_y, z=3)
            walker_transform = carla.Transform(walker_spawn_location, carla.Rotation(0, 90, 0))
            walker_bp = random.choice(self.blueprint_library.filter('*walker.*'))
            self.walker = self.world.try_spawn_actor(walker_bp, walker_transform)
            if self.walker is not None:
                self.walker.apply_control(carla.WalkerControl(
                    direction=carla.Vector3D(x=random.uniform(0.3, 0.5), y=random.uniform(-1, 1), z=0),
                    speed=1))
                walker_spawned = True
            else:
                walker_attempts += 1
                print(f"Attempt {walker_attempts} failed for walker.")
        if not walker_spawned:
            print("Could not spawn walker without collision.")
        self.world.tick()

        # SPECTATOR

        spectator = self.world.get_spectator()
        if self.parked_vehicle is not None:
            transform = self.parked_vehicle.get_transform()
        else:
            transform = self.player.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(y=-10,z=28.5), carla.Rotation(pitch=-90)))
        self.world.tick()

        # CONTROLLER

        self.control_count = 0
        if self.control_mode == "PID":
            self.controller = PIDController.Controller()



    def get_observation(self):

         # EGO information
        velocity_vec = self.player.get_velocity()
        current_transform = self.player.get_transform()
        current_location = current_transform.location
        current_roration = current_transform.rotation
        current_x = current_location.x
        current_y = current_location.y
        current_yaw = wrap_angle(current_roration.yaw)
        current_speed = math.sqrt(velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)

        current_steer = self.steer


        acceleration_vec =  self.player.get_acceleration()
        current_acceleration = math.sqrt(acceleration_vec.x**2 + acceleration_vec.y**2 + acceleration_vec.z**2)
        sideslip = np.tanh(velocity_vec.x/np.abs(velocity_vec.y+0.1))

  

        self.save_list.append([self.episode_counter,  self.desired_speed, self.last_v, self.ttc_trigger, self.distance_parked, self.clock.get_time(), current_x, current_y, current_speed, current_acceleration, 
                               acceleration_vec.x, acceleration_vec.y, sideslip, current_yaw, current_steer])

