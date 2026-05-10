"""
RL controller node. Loads a Stable-Baselines3 model and publishes actuator force.

ROS params:
  ~model_path    string  REQUIRED - path to .zip file
  ~algo          string  default: 'sac'
  ~deterministic bool    default: true
"""
import sys
from pathlib import Path

_core_path = Path(__file__).parents[3] / 'src' / 'quarter_car_core'
if _core_path.exists():
    sys.path.insert(0, str(_core_path))

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray

F_MAX = 10_000.0  # N


class RLNode(Node):
    def __init__(self):
        super().__init__('rl_node')
        self.declare_parameter('model_path',    '')
        self.declare_parameter('algo',          'sac')
        self.declare_parameter('deterministic', True)

        model_path = self.get_parameter('model_path').value
        algo       = self.get_parameter('algo').value
        self._det  = self.get_parameter('deterministic').value

        if not model_path:
            self.get_logger().error('model_path parameter is required')
            raise SystemExit(1)

        from stable_baselines3 import SAC, TD3, PPO
        cls = {'sac': SAC, 'td3': TD3, 'ppo': PPO}[algo.lower()]
        self._model = cls.load(model_path)
        self.get_logger().info(f'Loaded {algo} model from {model_path}')

        self._pub = self.create_publisher(Float64, '/actuator_force', 10)
        self.create_subscription(
            Float64MultiArray, '/car/state', self._cb_state, 10)

    def _cb_state(self, msg: Float64MultiArray):
        if len(msg.data) != 8:
            return
        obs = np.array(msg.data, dtype=np.float32)
        action, _ = self._model.predict(obs, deterministic=self._det)
        F_act = float(np.clip(action[0], -1.0, 1.0)) * F_MAX
        self._pub.publish(Float64(data=F_act))


def main(args=None):
    rclpy.init(args=args)
    node = RLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
