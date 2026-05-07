"""
sim_node — 50 Hz physics simulation node.

Publishes:
  /car/sprung_mass_pose   geometry_msgs/PoseStamped  (z_s)
  /car/wheel_pose         geometry_msgs/PoseStamped  (z_u)
  /car/road_height        std_msgs/Float64            (z_r)
  /car/state              std_msgs/Float64MultiArray  (8-dim obs)
  /car/acceleration       std_msgs/Float64            (z_s_ddot)
  /car/reward             std_msgs/Float64
  /car/comfort_score      std_msgs/Float64

Subscribes:
  /actuator_force         std_msgs/Float64

ROS params:
  ~road_profile   string  default: 'speed_bump'
  ~vehicle_speed  float   default: 10.0
  ~passive        bool    default: false
"""
import sys
from pathlib import Path

# quarter_car_core is pip-installed editable; this fallback handles dev without install
_core_path = Path(__file__).parents[3] / 'src' / 'quarter_car_core'
if _core_path.exists():
    sys.path.insert(0, str(_core_path))

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray
from geometry_msgs.msg import PoseStamped
import numpy as np

from quarter_car_core.ode_model import QuarterCarODE
from quarter_car_core.road_generator import RoadGenerator
from quarter_car_core.reward import RewardConfig, compute_reward

DT = 0.02  # 50 Hz


class SimNode(Node):
    def __init__(self):
        super().__init__('sim_node')
        self.declare_parameter('road_profile',  'speed_bump')
        self.declare_parameter('vehicle_speed', 10.0)
        self.declare_parameter('passive',       False)

        profile       = self.get_parameter('road_profile').value
        speed         = self.get_parameter('vehicle_speed').value
        self._passive = self.get_parameter('passive').value

        self._ode     = QuarterCarODE()
        self._road    = RoadGenerator(profile, speed)
        self._rcfg    = RewardConfig()
        self._state   = self._ode.reset()
        self._t       = 0.0
        self._F_act   = 0.0
        self._accel_sq = 0.0
        self._n_steps  = 0

        self._pub_sp  = self.create_publisher(PoseStamped,        '/car/sprung_mass_pose', 10)
        self._pub_wh  = self.create_publisher(PoseStamped,        '/car/wheel_pose',       10)
        self._pub_rh  = self.create_publisher(Float64,            '/car/road_height',      10)
        self._pub_st  = self.create_publisher(Float64MultiArray,  '/car/state',            10)
        self._pub_acc = self.create_publisher(Float64,            '/car/acceleration',     10)
        self._pub_rew = self.create_publisher(Float64,            '/car/reward',           10)
        self._pub_cs  = self.create_publisher(Float64,            '/car/comfort_score',    10)

        self.create_subscription(Float64, '/actuator_force', self._cb_force, 10)
        self.create_timer(DT, self._tick)
        self.get_logger().info(
            f'sim_node started: profile={profile}, passive={self._passive}')

    def _cb_force(self, msg: Float64):
        if not self._passive:
            self._F_act = msg.data

    def _tick(self):
        z_r = self._road.get_height(self._t)
        F   = 0.0 if self._passive else self._F_act

        new_state, z_s_ddot = self._ode.step(self._state, F, z_r, DT)
        self._state  = new_state
        self._t     += DT
        self._n_steps += 1

        z_s, z_s_dot, z_u, z_u_dot = self._state
        z_r_dot = self._road.get_height_dot(self._t)
        travel  = z_s - z_u
        tyre    = z_u - z_r

        self._accel_sq += z_s_ddot ** 2
        rms_accel = np.sqrt(self._accel_sq / self._n_steps)
        comfort   = max(0.0, 1.0 - rms_accel / self._rcfg.a_limit)
        reward    = compute_reward(z_s_ddot, travel, tyre, F, self._rcfg)

        now   = self.get_clock().now().to_msg()
        obs_8 = [z_s, z_s_dot, z_u, z_u_dot, z_r, z_r_dot, travel, tyre]

        def _pose(z_val: float) -> PoseStamped:
            p = PoseStamped()
            p.header.stamp    = now
            p.header.frame_id = 'map'
            p.pose.position.z = float(z_val)
            p.pose.orientation.w = 1.0
            return p

        self._pub_sp.publish(_pose(z_s))
        self._pub_wh.publish(_pose(z_u))
        self._pub_rh.publish(Float64(data=float(z_r)))
        ma = Float64MultiArray()
        ma.data = [float(v) for v in obs_8]
        self._pub_st.publish(ma)
        self._pub_acc.publish(Float64(data=float(z_s_ddot)))
        self._pub_rew.publish(Float64(data=float(reward)))
        self._pub_cs.publish(Float64(data=float(comfort)))


def main(args=None):
    rclpy.init(args=args)
    node = SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
