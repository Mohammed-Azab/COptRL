"""LQR controller node — stub, publishes 0.0 to /actuator_force."""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class LQRNode(Node):
    def __init__(self):
        super().__init__('lqr_node')
        self._pub = self.create_publisher(Float64, '/actuator_force', 10)
        self.create_timer(0.02, self._tick)
        self.get_logger().info('LQR stub — passive mode')

    def _tick(self):
        self._pub.publish(Float64(data=0.0))


def main(args=None):
    rclpy.init(args=args)
    node = LQRNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
