#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <iostream>

struct square
{
    __host__ __device__
    float operator()(const float& x) const
    {
        return x * x;
    }
};

struct add_two
{
    __host__ __device__
    float operator()(const thrust::tuple<float, float>& t) const
    {
        return thrust::get<0>(t) + thrust::get<1>(t);
    }
};

int main()
{
    // Create host vectors
    thrust::host_vector<float> h_vec1(10);
    thrust::host_vector<float> h_vec2(10);

    // Initialize host vectors  
    for (int i = 0; i < 10; ++i)
    {
        h_vec1[i] = static_cast<float>(i) + 1.0f; // Fill with values 1.0, 2.0, ..., 10.0
        h_vec2[i] = static_cast<float>(i + 1) * 2.0f; // Fill with values 2.0, 4.0, ..., 20.0
    }

    // Transfer data to the device
    thrust::device_vector<float> d_vec1 = h_vec1;
    thrust::device_vector<float> d_vec2 = h_vec2;

    // Create a device vector to hold the results
    thrust::device_vector<float> d_squared(10);
    auto square_itor = thrust::make_transform_iterator(d_vec1.begin(), square());
    // Apply the square operation to d_vec1
    thrust::copy(square_itor, square_itor + 10, d_squared.begin());

    // Print the squared results
    std::cout << "Squared values of d_vec1:" << std::endl;
    thrust::copy(d_squared.begin(), d_squared.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    // Create a device vector to hold the addition results
    thrust::device_vector<float> d_result(10);

    // Apply the add_two operation to d_vec1 and d_vec2
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(d_vec1.begin(), d_vec2.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(d_vec1.end(), d_vec2.end())),
        d_result.begin(),
        add_two()
    );

    // Print the addition results
    std::cout << "Sum of corresponding elements in d_vec1 and d_vec2:" << std::endl;
    thrust::copy(d_result.begin(), d_result.end(), std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    return 0;
}
