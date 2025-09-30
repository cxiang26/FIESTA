#ifndef VISUALIZATION_MSGS_MARKER_H_
#define VISUALIZATION_MSGS_MARKER_H_
#include <vector>
#include "std_msgs/ColorRGBA.h"
#include "geometry_msgs/Point.h"
namespace visualization_msgs {
struct Marker {
  struct Header { const char* frame_id; } header; // minimal
  int id;
  int type;
  int action;
  struct Scale { double x, y, z; } scale;
  struct Pose { struct Orientation { double w,x,y,z; } orientation; } pose;
  std::vector<geometry_msgs::Point> points;
  std::vector<std_msgs::ColorRGBA> colors;
  static const int POINTS = 8;
  static const int MODIFY = 0;
};
}
#endif // VISUALIZATION_MSGS_MARKER_H_ 