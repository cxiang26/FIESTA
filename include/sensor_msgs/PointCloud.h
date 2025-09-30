#ifndef SENSOR_MSGS_POINTCLOUD_H_
#define SENSOR_MSGS_POINTCLOUD_H_
#include <vector>
#include "geometry_msgs/Point32.h"
namespace sensor_msgs {
struct PointCloud {
  struct Header { const char* frame_id; } header;
  std::vector<geometry_msgs::Point32> points;
};
}
#endif // SENSOR_MSGS_POINTCLOUD_H_ 