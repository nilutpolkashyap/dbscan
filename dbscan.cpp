#include "dbscan.hpp"
#include <cstddef>
#include <nanoflann/nanoflann.hpp>
#include <type_traits>
#include <vector>
#include <algorithm>

// Helper function to get the coordinate of a 2D point given the dimension (0 for x, 1 for y)
inline float get_pt(const Point2& p, std::size_t dim)
{
    if (dim == 0) return p.x;
    return p.y;
}

// Helper function to get the coordinate of a 3D point given the dimension (0 for x, 1 for y, 2 for z)
inline float get_pt(const Point3& p, std::size_t dim)
{
    if (dim == 0) return p.x;
    if (dim == 1) return p.y;
    return p.z;
}

// Adaptor class for interfacing with the KD-tree implementation
template<typename Point>
struct Adaptor
{
    const std::span<const Point>& points;
    Adaptor(const std::span<const Point>& points) : points(points) {}

    // Must return the number of data points
    inline std::size_t kdtree_get_point_count() const { return points.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline float kdtree_get_pt(const std::size_t idx, const std::size_t dim) const
    {
        return get_pt(points[idx], dim);
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }

    // Return a pointer to the x coordinate of the idx'th point
    auto const * elem_ptr(const std::size_t idx) const
    {
        return &points[idx].x;
    }
};

// Function to sort clusters by their point indices
void sort_clusters(std::vector<std::vector<size_t>>& clusters)
{
    for (auto& cluster : clusters)
    {
        std::sort(cluster.begin(), cluster.end());
    }
}

template<int n_cols, typename Adaptor>
std::vector<std::vector<size_t>> dbscan_impl(const Adaptor& adapt, float eps, int min_pts)
{
    // Squaring epsilon for distance comparison
    eps *= eps;
    
    using namespace nanoflann;
    using my_kd_tree_t = KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<float, Adaptor>, Adaptor, n_cols>;

    // Building the KD-tree index
    auto index = my_kd_tree_t(n_cols, adapt, KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();

    const auto n_points = adapt.kdtree_get_point_count();
    auto visited = std::vector<bool>(n_points);
    auto clusters = std::vector<std::vector<size_t>>();
    auto matches = std::vector<std::pair<size_t, float>>();
    auto sub_matches = std::vector<std::pair<size_t, float>>();

    for (size_t i = 0; i < n_points; i++)
    {
        if (visited[i]) continue;

        // Radius search for neighbors within epsilon distance
        index.radiusSearch(adapt.elem_ptr(i), eps, matches, SearchParams(32, 0.f, false));
        if (matches.size() < static_cast<size_t>(min_pts)) continue;
        visited[i] = true;

        // Creating a new cluster and adding the core point
        auto cluster = std::vector({i});

        while (!matches.empty())
        {
            auto nb_idx = matches.back().first;
            matches.pop_back();
            if (visited[nb_idx]) continue;
            visited[nb_idx] = true;

            // Radius search for neighbors of the neighbor
            index.radiusSearch(adapt.elem_ptr(nb_idx), eps, sub_matches, SearchParams(32, 0.f, false));

            if (sub_matches.size() >= static_cast<size_t>(min_pts))
            {
                std::copy(sub_matches.begin(), sub_matches.end(), std::back_inserter(matches));
            }
            cluster.push_back(nb_idx);
        }
        clusters.emplace_back(std::move(cluster));
    }
    sort_clusters(clusters);
    return clusters;
}

// DBSCAN class constructor
DBSCAN::DBSCAN(float eps, int min_pts)
    : eps_(eps), min_pts_(min_pts)
{}

// DBSCAN run method for 2D points
std::vector<std::vector<size_t>> DBSCAN::run(const std::span<const Point2>& data)
{
    const auto adapt = Adaptor<Point2>(data);
    return dbscan_impl<2>(adapt, eps_, min_pts_);
}

// DBSCAN run method for 3D points
std::vector<std::vector<size_t>> DBSCAN::run(const std::span<const Point3>& data)
{
    const auto adapt = Adaptor<Point3>(data);
    return dbscan_impl<3>(adapt, eps_, min_pts_);
}
