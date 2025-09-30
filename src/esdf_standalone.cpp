#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

#include <Eigen/Eigen>
#include "ESDFMap.h"
#include "raycast.h"

// OpenCV for reading depth image
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace Eigen;

struct Intrinsics {
  double fx{0}, fy{0}, cx{0}, cy{0};
};

static bool LoadIntrinsics(const string &path, Intrinsics &K) {
  ifstream fin(path);
  if (!fin.is_open()) return false;
  string line;
  while (getline(fin, line)) {
    if (line.empty()) continue;
    string key; char colon;
    double val;
    stringstream ss(line);
    ss >> key >> colon >> val;
    if (key.find("cx") != string::npos) K.cx = val;
    else if (key.find("cy") != string::npos) K.cy = val;
    else if (key.find("fx") != string::npos) K.fx = val;
    else if (key.find("fy") != string::npos) K.fy = val;
  }
  return (K.fx > 0 && K.fy > 0);
}

static bool LoadPoseTxTyTzQxQyQzQw(const string &path, Vector3d &t, Quaterniond &q) {
  ifstream fin(path);
  if (!fin.is_open()) return false;
  double tx, ty, tz, qx, qy, qz, qw;
  fin >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
  if (!fin.good()) return false;
  t = Vector3d(tx, ty, tz);
  q = Quaterniond(qw, qx, qy, qz);
  q.normalize();
  return true;
}

static inline bool IsFiniteDepth(uint16_t d) {
  return d > 0 && d < 65500;
}

int main(int argc, char **argv) {
  string base = "/home/xcq/projects/FIESTA/examples";
  string depth_path = base + "/depth_000.png";
  string intr_path = base + "/intrinsic_000.txt";
  string pose_path  = base + "/pose_000.txt";

  Intrinsics K;
  if (!LoadIntrinsics(intr_path, K)) {
    cerr << "Failed to load intrinsics from " << intr_path << endl;
    return 1;
  }
  Vector3d t_w_c; Quaterniond q_w_c;
  if (!LoadPoseTxTyTzQxQyQzQw(pose_path, t_w_c, q_w_c)) {
    cerr << "Failed to load pose from " << pose_path << endl;
    return 1;
  }

  // Load depth (assume 16UC1, unit: meters or millimeters; we will treat as meters if values < 20, else meters = d/1000)
  cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
  if (depth.empty()) {
    cerr << "Failed to load depth image from " << depth_path << endl;
    return 1;
  }
  if (depth.type() != CV_16UC1) {
    cerr << "Depth image is not 16UC1; type=" << depth.type() << endl;
  }

  // ESDF map configuration (standalone): choose a bounding box around camera
  double resolution = 0.1; // meters per voxel
  Vector3d origin(.0, .0, -2.); // world origin of voxel grid
  Vector3d map_size(5.0, 5.0, 5.0); // meters

#ifndef HASH_TABLE
  fiesta::ESDFMap esdf(origin, resolution, map_size);
#else
  fiesta::ESDFMap esdf(origin, resolution, 1000000);
#endif

#ifdef PROBABILISTIC
  esdf.SetParameters(0.70, 0.35, 0.12, 0.97, 0.80);
#endif

  // Raycast bounds in voxel index space: [0, grid_size)
  Vector3i grid_size;
  for (int i = 0; i < 3; ++i) grid_size(i) = static_cast<int>(std::ceil(map_size(i) / resolution));
  Vector3d min_vox(0, 0, 0);
  Vector3d max_vox(grid_size.cast<double>());

  // Camera origin in world and in voxel coordinates
  Vector3d Ow = t_w_c;
  auto WorldToVoxel = [&](const Vector3d &p) {
    return (p - origin) / resolution;
  };
  Vector3d Ov = WorldToVoxel(Ow);

  // Iterate depth image (subsample to speed up)
  int stride = 4;
  int width = depth.cols;
  int height = depth.rows;
  int num_rays = 0;

  for (int v = 0; v < height; v += stride) {
    const uint16_t *row = depth.ptr<uint16_t>(v);
    for (int u = 0; u < width; u += stride) {
      uint16_t d = row[u];
      if (!IsFiniteDepth(d)) continue;

      double depth_m = static_cast<double>(d);
      if (depth_m > 50.0) depth_m *= 0.001; // treat as millimeters → meters

      // Back-project to camera frame
      double x = (u - K.cx) / K.fx;
      double y = (v - K.cy) / K.fy;
      Vector3d Pc = Vector3d(x * depth_m, y * depth_m, depth_m);

      // Transform to world
      Vector3d Pw = q_w_c * Pc + t_w_c;

      // Convert to voxel coords
      Vector3d Pv = WorldToVoxel(Pw);

      // Raycast from Ov to Pv
      std::vector<Vector3d> voxels;
      Raycast(Ov, Pv, min_vox, max_vox, &voxels);
      if (voxels.empty()) continue;

      // Mark free space along ray, and occupied at the last voxel
      for (size_t i = 0; i + 1 < voxels.size(); ++i) {
        Vector3i vi(static_cast<int>(voxels[i].x()), static_cast<int>(voxels[i].y()), static_cast<int>(voxels[i].z()));
        esdf.SetOccupancy(vi, 0);
      }
      Vector3d vend = voxels.back();
      Vector3i vi_end(static_cast<int>(vend.x()), static_cast<int>(vend.y()), static_cast<int>(vend.z()));
      esdf.SetOccupancy(vi_end, 1);

      ++num_rays;
    }
  }

  // Fuse occupancy (probabilistic) and update ESDF
  esdf.UpdateOccupancy(true);
  esdf.UpdateESDF();

  // Query and print a small grid of distances near the camera
  cout << "Processed rays: " << num_rays << endl;
  cout << "Sample distances (meters) around camera origin:" << endl;
  for (int dz = -2; dz <= 2; ++dz) {
    for (int dy = -2; dy <= 2; ++dy) {
      for (int dx = -2; dx <= 2; ++dx) {
        Vector3d p = Ow + Vector3d(dx * 0.2, dy * 0.2, dz * 0.2);
        double dist = esdf.GetDistance(p);
        cout << dist << (dx == 2 ? '\n' : ' ');
      }
    }
  }

  // Export a slice for quick sanity (z slice index around camera)
  visualization_msgs::Marker marker;
  int slice = static_cast<int>(std::floor((Ow.z() - origin.z()) / resolution));
  slice = std::max(0, std::min(slice, grid_size(2) - 1));
  esdf.GetSliceMarker(marker, slice, 0, Vector4d(1, 1, 1, 1), 2.0);
  cout << "Slice points at z-index " << slice << ": " << marker.points.size() << endl;

  // Save occupancy as PLY point cloud (occupied voxels' centers)
  sensor_msgs::PointCloud cloud;
  esdf.GetPointCloud(cloud, 0, grid_size(2));
  {
    ofstream ply("/home/xcq/projects/FIESTA/build/occ_points.ply");
    ply << "ply\nformat ascii 1.0\n";
    ply << "element vertex " << cloud.points.size() << "\n";
    ply << "property float x\nproperty float y\nproperty float z\nend_header\n";
    for (const auto &p : cloud.points) {
      ply << p.x << ' ' << p.y << ' ' << p.z << '\n';
    }
    ply.close();
    cout << "Saved PLY: /home/xcq/projects/FIESTA/build/occ_points.ply (" << cloud.points.size() << " pts)\n";
  }

  // Save a 2D occupancy slice image at the chosen z-index
  {
    cv::Mat occ(grid_size(1), grid_size(0), CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < grid_size(1); ++y) {
      for (int x = 0; x < grid_size(0); ++x) {
        int occ_val = esdf.GetOccupancy(Vector3i(x, y, slice));
        if (occ_val == 1) occ.at<uint8_t>(y, x) = 255;
      }
    }
    string out_png = "/home/xcq/projects/FIESTA/build/occ_slice.png";
    cv::imwrite(out_png, occ);
    cout << "Saved PNG: " << out_png << " (slice z-index=" << slice << ")\n";
  }

  // Save ESDF 2D slice as color image (Jet colormap)
  {
    double max_vis_dist = 2.0; // meters
    cv::Mat dist_img(grid_size(1), grid_size(0), CV_32FC1, cv::Scalar(0));
    for (int y = 0; y < grid_size(1); ++y) {
      for (int x = 0; x < grid_size(0); ++x) {
        double d = esdf.GetDistance(Vector3i(x, y, slice));
        if (d >= 0 && d < 1e9) dist_img.at<float>(y, x) = static_cast<float>(std::min(d, max_vis_dist));
      }
    }
    cv::Mat norm8, color;
    dist_img.convertTo(norm8, CV_8UC1, 255.0 / max_vis_dist);
    cv::applyColorMap(norm8, color, cv::COLORMAP_JET);
    string out_png = "/home/xcq/projects/FIESTA/build/esdf_slice.png";
    cv::imwrite(out_png, color);
    cout << "Saved ESDF slice PNG: " << out_png << " (max_vis_dist=" << max_vis_dist << ")\n";
  }

  // Save ESDF slice as colored PLY using Marker points/colors
  {
    ofstream ply("/home/xcq/projects/FIESTA/build/esdf_slice_colored.ply");
    ply << "ply\nformat ascii 1.0\n";
    ply << "element vertex " << marker.points.size() << "\n";
    ply << "property float x\nproperty float y\nproperty float z\n";
    ply << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for (size_t i = 0; i < marker.points.size(); ++i) {
      const auto &p = marker.points[i];
      unsigned char r = 255, g = 255, b = 255;
      if (i < marker.colors.size()) {
        r = static_cast<unsigned char>(std::round(std::max(0.f, std::min(1.f, marker.colors[i].r)) * 255));
        g = static_cast<unsigned char>(std::round(std::max(0.f, std::min(1.f, marker.colors[i].g)) * 255));
        b = static_cast<unsigned char>(std::round(std::max(0.f, std::min(1.f, marker.colors[i].b)) * 255));
      }
      ply << p.x << ' ' << p.y << ' ' << p.z << ' ' << (int)r << ' ' << (int)g << ' ' << (int)b << "\n";
    }
    ply.close();
    cout << "Saved colored PLY slice: /home/xcq/projects/FIESTA/build/esdf_slice_colored.ply\n";
  }

  // Save full 3D ESDF volume as VTK Structured Points (legacy ASCII)
  {
    const int nx = grid_size(0), ny = grid_size(1), nz = grid_size(2);
    string out_vtk = "/home/xcq/projects/FIESTA/build/esdf_volume.vtk";
    ofstream vtk(out_vtk);
    vtk << "# vtk DataFile Version 3.0\n";
    vtk << "ESDF volume\n";
    vtk << "ASCII\n";
    vtk << "DATASET STRUCTURED_POINTS\n";
    vtk << "DIMENSIONS " << nx << ' ' << ny << ' ' << nz << "\n";
    // origin at voxel centers: origin + 0.5*res
    vtk << "ORIGIN " << (origin.x() + 0.5 * resolution) << ' ' << (origin.y() + 0.5 * resolution) << ' ' << (origin.z() + 0.5 * resolution) << "\n";
    vtk << "SPACING " << resolution << ' ' << resolution << ' ' << resolution << "\n";
    vtk << "POINT_DATA " << (nx * ny * nz) << "\n";
    vtk << "SCALARS distance float 1\n";
    vtk << "LOOKUP_TABLE default\n";
    for (int z = 0; z < nz; ++z) {
      for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
          double d = esdf.GetDistance(Vector3i(x, y, z));
          if (d > 9e3) d = -1.0; // mark unknown as -1
          vtk << static_cast<float>(d) << '\n';
        }
      }
    }
    vtk.close();
    cout << "Saved ESDF volume VTK: " << out_vtk << "\n";
  }

  return 0;
} 