#pragma once

#include <cassert>
#include <cstddef>
#include <span>
#include <vector>
#include <cstdlib>

// Structure to represent a 2D point
struct Point2
{
    float x, y;
};

// Structure to represent a 3D point
struct Point3
{
    float x, y, z;
};

// DBSCAN class definition
class DBSCAN
{
public:
    // Constructor to initialize the DBSCAN parameters
    DBSCAN(float eps, int min_pts);

    // Method to run DBSCAN on 2D data
    std::vector<std::vector<size_t>> run(const std::span<const Point2>& data);

    // Method to run DBSCAN on 3D data
    std::vector<std::vector<size_t>> run(const std::span<const Point3>& data);

private:
    float eps_;   // Epsilon value for neighborhood radius
    int min_pts_; // Minimum number of points to form a cluster

    // Template method to run DBSCAN on generic data points (2D or 3D)
    template<typename Point>
    std::vector<std::vector<size_t>> run_impl(const std::span<const Point>& data);
};
