"""
gazebo_bridge_node — converts z_s to Gazebo joint command and TF transforms.

Subscribes:
  /car/sprung_mass_pose  PoseStamped   (z_s from sim_node)
  /car/road_height       Float64

Publishes to Gazebo via ros_gz_bridge:
  /model/quarter_car/joint/chassis_joint/cmd_pos  Float64  (z_s + 0.15)

Publishes TF:
  map -> base_link  (static, identity)
  base_link -> chassis_link  (z = z_s + 0.15, x advancing at 10 m/s)
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

CHASSIS_Z_OFFSET = 0.15   # m
CAR_SPEED        = 10.0   # m/s — car moves forward in Gazebo world
CAR_START_X      = 0.0    # m


class GazeboBridgeNode(Node):
    def __init__(self):
        super().__init__('gazebo_bridge_node')
        self._z_s = 0.0
        self._t   = 0.0

        self._br        = TransformBroadcaster(self)
        self._static_br = StaticTransformBroadcaster(self)
        self._pub_gz    = self.create_publisher(
            Float64, '/model/quarter_car/joint/chassis_joint/cmd_pos', 10)

        self.create_subscription(
            PoseStamped, '/car/sprung_mass_pose', self._cb_sprung, 10)
        self.create_subscription(
            Float64, '/car/road_height', self._cb_road, 10)

        self._publish_static_tf()
        self.create_timer(0.02, self._publish_dynamic_tf)  # 50 Hz

    def _publish_static_tf(self):
        st = TransformStamped()
        st.header.stamp    = self.get_clock().now().to_msg()
        st.header.frame_id = 'map'
        st.child_frame_id  = 'base_link'
        st.transform.rotation.w = 1.0
        self._static_br.sendTransform(st)

    def _cb_sprung(self, msg: PoseStamped):
        self._z_s = msg.pose.position.z

    def _cb_road(self, msg: Float64):
        pass  # available for future use

    def _publish_dynamic_tf(self):
        self._t += 0.02
        x_car = CAR_START_X + CAR_SPEED * self._t
        z_cmd = self._z_s + CHASSIS_Z_OFFSET

        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id  = 'chassis_link'
        t.transform.translation.x = float(x_car)
        t.transform.translation.z = float(z_cmd)
        t.transform.rotation.w    = 1.0
        self._br.sendTransform(t)

        self._pub_gz.publish(Float64(data=float(z_cmd)))


def main(args=None):
    rclpy.init(args=args)
    node = GazeboBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
