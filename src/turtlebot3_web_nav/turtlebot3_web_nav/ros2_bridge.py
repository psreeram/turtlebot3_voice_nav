import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
import sys

class ROS2Bridge(Node):
    def __init__(self):
        super().__init__('ros2_bridge')
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def set_goal(self, x, y, theta):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.pose.position.x = float(x)
        goal_pose.pose.position.y = float(y)
        goal_pose.pose.orientation.z = float(theta)
        goal_pose.pose.orientation.w = 1.0

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.nav_to_pose_client.wait_for_server()
        self.send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Navigation succeeded!')
        else:
            self.get_logger().info('Navigation failed with status: {0}'.format(status))

        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    ros2_bridge = ROS2Bridge()

    if len(sys.argv) > 1:
        if sys.argv[1] == 'set_goal' and len(sys.argv) == 5:
            ros2_bridge.set_goal(sys.argv[2], sys.argv[3], sys.argv[4])
        elif sys.argv[1] == 'start_navigation':
            pass  # Navigation starts automatically after setting the goal
        else:
            print("Invalid arguments")
            rclpy.shutdown()
            return

    rclpy.spin(ros2_bridge)

if __name__ == '__main__':
    main()