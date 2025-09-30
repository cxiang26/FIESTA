#ifndef PCL_KDTREE_KDTREE_FLANN_H_
#define PCL_KDTREE_KDTREE_FLANN_H_
#include <vector>
#include <memory>
namespace pcl {
struct PointXYZ {
  float x{0}, y{0}, z{0};
  PointXYZ() = default;
  PointXYZ(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}
};

template<typename PointT = PointXYZ>
struct PointCloud {
  using Ptr = std::shared_ptr<PointCloud<PointT>>;
  unsigned int width{0};
  unsigned int height{1};
  std::vector<PointT> points;
};

template<typename PointT = PointXYZ>
class KdTreeFLANN {
 public:
  void setInputCloud(const typename PointCloud<PointT>::Ptr&) {}
  int nearestKSearch(const PointT&, int, std::vector<int>&, std::vector<float>&) { return 0; }
};
}
#endif // PCL_KDTREE_KDTREE_FLANN_H_ 