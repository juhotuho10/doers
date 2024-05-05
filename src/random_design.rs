use ndarray_rand::rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use ndarray_rand::rand_distr::{Distribution, Uniform};
use ndarray_rand::RandomExt;

use ndarray::stack;
use ndarray::{s, Array, Array1, Array2, Axis, Zip};
use ndarray_rand::rand::{thread_rng, Rng};
use std::vec;
/*
This code was originally published by the following individuals for use with Scilab:
    Copyright (C) 2012 - 2013 - Michael Baudin
    Copyright (C) 2012 - Maria Christopoulou
    Copyright (C) 2010 - 2011 - INRIA - Michael Baudin
    Copyright (C) 2009 - Yann Collette
    Copyright (C) 2009 - CEA - Jean-Marc Martinez
    website: https://atoms.scilab.org/toolboxes/scidoe/0.4.1

Converted to python and worked on by:
    Copyright (c) 2014, Abraham D. Lee
    git repo: https://github.com/tisimst/pyDOE

    Copyright (c) 2018, Rickard Sjögren & Daniel Svensson
    git repo: https://github.com/clicumu/pyDOE2

Converted to Rust and worked on by:
    Copyright (c) 2024, Juho Naatula
    git repo: https://github.com/juhotuho10/doers
*/

/**
Generates a classic latin-hypercube design.

# Parameters

- `n`: `usize`
  The number of factors to generate samples for.

- `samples`: `usize`
  The number of samples to generate for each factor.

- `random_state`: `u64`
  Seed-number that controls the random draws.

# Returns

- `H`: `Array2<f32>`
  `n` by `samples` design matrix where the columns are random but tend to have values that are somewhat equally spaced

# Example

A 4-sample design:
```rust
use doers::random_design::lhs_classic;
let n = 4;
let samples = 4;
let random_state = 42;
let example_array = lhs_classic(n, samples, random_state);
// resulting Array2:
// [[0.32606143,  0.5795995,  0.773937,     0.88865906],
//  [0.9321394,   0.8648242,  0.0036229491, 0.09710106],
//  [0.039839655, 0.28775924, 0.30786952,   0.4689691],
//  [0.5738912,   0.13481694, 0.5605331,    0.6929374]]
```

# Guarantees

None. But columns tend to be equally spaced since in the case of samples = 4
Array1s with uniform distribution are scaled with  another array: [0.25, 0.5, 0.75, 1]
and then shuffled
*/
#[allow(dead_code)]
pub fn lhs_classic(n: usize, samples: usize, random_state: u64) -> Array2<f32> {
    // Generate a random array using `rng`
    let mut rng = SmallRng::seed_from_u64(random_state);

    let array_shape = (n, samples);
    let u = Array::random_using(array_shape, Uniform::new(0., 1.), &mut rng);

    let cut = Array::linspace(0., 1., samples + 1);

    let a = cut.slice(s![..samples]);
    let b = cut.slice(s![1..samples + 1]);
    let mut h_array: Array2<f32> = Array::zeros(array_shape);

    for i in 0..n {
        let mut mutable = h_array.slice_mut(s![i, ..]);
        let u_slice = u.slice(s![i, ..]);
        let assignable = &u_slice * (&b - &a) + a;
        mutable.assign(&assignable);
    }

    for i in 0..n {
        let mut row = h_array.row_mut(i);
        row.as_slice_mut().unwrap().shuffle(&mut thread_rng());
    }

    h_array = h_array.t().to_owned();

    h_array
}

/**
Generates a latin-hypercube design with equal spacing between each value.

# Parameters

- `n`: `usize`
  The number of factors to generate samples for.

- `samples`: `usize`
  The number of samples to generate for each factor.

- `random_state`: `u64`
  Seed-number that controls the random draws.

# Returns

- `H`: `Array2<f32>`
  `n` by `samples` design matrix where the columns are random but tend to have values that are somewhat equally spaced

# Example

A 4-sample design:
```rust
use doers::random_design::lhs_centered;
let n = 4;
let samples = 4;
let random_state = 42;
let example_array = lhs_centered(n, samples, random_state);
// resulting Array2:
// [[0.125, 0.625, 0.875, 0.375],
//  [0.875, 0.125, 0.375, 0.125],
//  [0.625, 0.375, 0.125, 0.875],
//  [0.375, 0.875, 0.625, 0.625]]
```

# Guarantees

guaranteed to be equally spaced, startin from 0.5/`samples` and jumping 1/`samples` per sample

in case of `samples` = 4
starting from 0.125 and continuing in 0.25 steps
*/
#[allow(dead_code)]
pub fn lhs_centered(n: usize, samples: usize, random_state: u64) -> Array2<f32> {
    // Generate a random array using `rng`
    let mut rng = SmallRng::seed_from_u64(random_state);

    let array_shape = (samples, n);

    let mut cut: Array1<f64> = Array::linspace(0., 1., samples + 1);

    cut = cut.mapv(|x: f64| (x * 100.).round() / 100.); // rounding the variables to 0.01

    let a = cut.slice(s![..samples]);
    let b = cut.slice(s![1..samples + 1]);
    let mut center = ((&a + &b) / 2.).to_vec();
    let mut h_array = Array::zeros(array_shape);

    for i in 0..n {
        center.shuffle(&mut rng);
        let center_shuffled = Array::from_vec(center.clone());
        h_array.column_mut(i).assign(&center_shuffled);
    }

    h_array.mapv(|x| x as f32)
}

/**
Generates and iterates over classic latin-hypercube design to make it more equally spaced.

# Parameters

- `n`: `usize`
  The number of factors to generate samples for.

- `samples`: `usize`
  The number of samples to generate for each factor.

- `random_state`: `u64`
  Seed-number that controls the random draws.

- `iterations`: `u16`
  The number of iterations the function tries to maximize the distances.

# Returns

- `H`: `Array2<f32>`
  `n` by `samples` design matrix where the columns are random but tend to have values that are somewhat equally spaced

# Example

A 4-sample design:
```rust
use doers::random_design::lhs_maximin;
let n = 4;
let samples = 4;
let random_state = 42;
let iterations = 100;
let example_array = lhs_maximin(n, samples, random_state, iterations);
// resulting Array2:
// [[0.87440515, 0.4860021,  0.87007785,  0.43068066],
//  [0.40560502, 0.0374375,  0.3437121,   0.73863614],
//  [0.70085746, 0.63299274, 0.022075802, 0.23332724],
//  [0.07169747, 0.93537205, 0.69658417,  0.87581944]]
```

# Guarantees

None. But columns tend to be equally spaced since that is what the function tries to iterate over
*/
#[allow(dead_code)]
pub fn lhs_maximin(n: usize, samples: usize, random_state: u64, iterations: u16) -> Array2<f32> {
    let mut max_dist = 0.;
    let mut h_array: Array2<f32> = Array2::from_elem((n, samples), 0.);
    let mut rng = SmallRng::seed_from_u64(random_state);

    for _ in 0..iterations {
        let random_int = rng.gen_range(0..u64::MAX);
        let h_candidate = lhs_classic(n, samples, random_int);
        let dist_array = pairwise_euclidean_dist(&h_candidate).to_owned();
        let min_dist = *dist_array
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        if min_dist > max_dist {
            max_dist = min_dist;
            h_array = h_candidate;
        }
    }
    h_array
}

/**
Generates and iterates over classic latin-hypercube design to make it less correlated.

# Parameters

- `n`: `usize`
  The number of factors to generate samples for.

- `samples`: `usize`
  The number of samples to generate for each factor.

- `random_state`: `u64`
  Seed-number that controls the random draws.

- `iterations`: `u16`
  The number of iterations the function tries to maximize the distances.

# Returns

- `H`: `Array2<f32>`
  `n` by `samples` design matrix where the columns are random but tend to have values that are somewhat equally spaced

# Example

A 4-sample design:
```rust
use doers::random_design::lhs_correlate;
let n = 4;
let samples = 4;
let random_state = 42;
let iterations = 100;
let example_array = lhs_correlate(n, samples, random_state, iterations);
// resulting Array2:
// [[0.0793857,  0.6603068,   0.4616545,  0.03551933],
//  [0.49591884, 0.93893766,  0.8617085,  0.5154473],
//  [0.6710943,  0.075240016, 0.63285625, 0.9130845],
//  [0.89382416, 0.40597996,  0.14301169, 0.4802417]]
```

# Guarantees

None. Tries to aim at the design being more chaotic since the correlation of the design is minimized through iteration
*/
#[allow(dead_code)]
pub fn lhs_correlate(n: usize, samples: usize, random_state: u64, iterations: u16) -> Array2<f32> {
    let mut mincorr: f32 = f32::INFINITY;
    let mut h_array = Array2::<f32>::zeros((samples, n));
    let mut rng = SmallRng::seed_from_u64(random_state);

    for _ in 0..iterations {
        let random_int = rng.gen_range(0..u64::MAX);
        let h_candidate = lhs_classic(n, samples, random_int);
        let corr: Array2<f32> = corrcoef(&h_array.t().to_owned());
        let max_corr = corr
            .iter()
            .filter(|&&x| x != 1.0)
            .map(|x| x.abs())
            .fold(f32::MIN, f32::max);

        if max_corr < mincorr {
            mincorr = max_abs_off_diagonal(&corr);
            h_array = h_candidate
        }
    }

    h_array
}

/**
Generates a classic latin-hypercube design.

# Parameters

- `n`: `usize`
  The number of factors to generate samples for.

- `samples`: `usize`
  The number of samples to generate for each factor.

- `random_state`: `u64`
  Seed-number that controls the random draws.

# Returns

- `H`: `Array2<f32>`
  `n` by `samples` design matrix where the columns are random but tend to have values that are somewhat equally spaced

# Example

A 4-sample design:
```rust
use doers::random_design::lhs_mu;
let n = 4;
let samples = 4;
let random_state = 42;
let example_array = lhs_mu(n, samples, random_state);
// resulting Array2:
// [[0.6042977,   0.48937836, 0.7415293,  0.5962739],
//  [0.097561955, 0.71106726, 0.15533936, 0.48371363],
//  [0.8184801,   0.04350221, 0.41811958, 0.8805161],
//  [0.42012796,  0.90002126, 0.8194874,  0.15655315]]
```

# Guarantees

Guarantees that in every column, there is a random value in equally spaced 1/samples sized ranges

In the case of samples = 4, there will be a value between 0.0 - 0.25, another between 0.25 - 0.5 etc. and the ranges have uniform distribution
*/
#[allow(dead_code)]
pub fn lhs_mu(n: usize, samples: usize, random_state: u64) -> Array2<f32> {
    let mut rng = SmallRng::seed_from_u64(random_state);

    let size = 5 * samples;

    let array_shape = (size, n);
    let mut rdpoints = Array::random_using(array_shape, Uniform::new(0., 1.), &mut rng);

    let mut d_ij: Array2<f32> = cdist_euclidean(&rdpoints, &rdpoints);

    for i in 0..size.min(n) {
        d_ij[[i, i]] = f32::NAN;
    }

    let mut index_rm: Array1<usize> = Array::zeros(size - samples);

    for i in 0..samples * 4 {
        let mut order = d_ij.clone();
        order = sort_array2_by_axis_with_nan_handling(order);
        let avg_dist: Vec<f32> = order
            .axis_iter(Axis(0))
            .map(|row| mean_of_first_two(row.as_slice().unwrap()))
            .collect();

        let min_l = argmin_ignore_nan(&avg_dist).expect("We should have min index");

        for j in 0..d_ij.ncols() {
            d_ij[[min_l, j]] = f32::NAN;
            d_ij[[j, min_l]] = f32::NAN;
        }

        index_rm[i] = min_l;
    }

    rdpoints = delete_rows(rdpoints, index_rm);
    let rank: Array2<usize> = argsort_axis0(&rdpoints);
    let mut h_array: Array2<f32> = Array2::zeros((samples, n));

    let mut distr: Uniform<f32>;

    for l in 0..samples {
        let low: f32 = l as f32 / samples as f32;
        let high: f32 = (l + 1) as f32 / samples as f32;
        distr = Uniform::new(low, high);

        Zip::from(&rank)
            .and(&mut h_array)
            .for_each(|&rank_val, h_val| {
                if rank_val == l {
                    *h_val = distr.sample(&mut rng);
                }
            });
    }
    h_array
}

// ##############################################################################################################
// ----------------------------------------------- Utilities ----------------------------------------------------
// ##############################################################################################################

/**
Calculates the Euclidean distance between two points in n-dimensional space.

The Euclidean distance is the straight-line distance between two points in Euclidean space. It is calculated as the square root of the sum of the squared differences between the corresponding elements of the two points.

# Parameters

- `a`: &Array1<f32>
    A one-dimensional array representing the coordinates of the first point in n-dimensional space. Each element in the array corresponds to a coordinate in a particular dimension.

- `b`: Array1<f32>
    A one-dimensional array representing the coordinates of the second point in n-dimensional space. The length of `b` must match the length of `a` to correctly calculate the distance.

# Returns

- `f32`
    The Euclidean distance between the two points as a floating-point number. The distance is non-negative and represents the "length" of the straight line connecting the two points in n-dimensional space.
*/
fn euclidean_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}
/**
Computes the pairwise Euclidean distances between rows in a 2D array.

Each row in the input array represents a point in n-dimensional space. This function calculates the Euclidean distance between every pair of points (rows) and returns a one-dimensional array containing these distances.

# Parameters

- `input`: &Array2<f32>
    A two-dimensional array where each row represents a point in n-dimensional space. The dimensions of the array are expected to be `[number_of_points, dimensions_of_each_point]`.

# Returns

- `Array1<f32>`
    A one-dimensional array of floating-point numbers, where each element is the Euclidean distance between a pair of points (rows) in the input array.
    The distances are listed in the order they were computed, which corresponds to a row-wise upper triangular traversal of the pairwise distance matrix, excluding the diagonal.
*/
fn pairwise_euclidean_dist(input: &Array2<f32>) -> Array1<f32> {
    let rows = input.nrows();
    let mut distances = Vec::new();

    for i in 0..rows {
        for j in i + 1..rows {
            let distance = euclidean_distance(&input.row(i).to_owned(), &input.row(j).to_owned());

            distances.push(distance);
        }
    }

    Array1::from(distances)
}

/**
Computes the Euclidean distances between each pair of the two collections of inputs.

The function takes two 2-dimensional arrays, `a` and `b`, each representing a collection of points in n-dimensional space.
It calculates the Euclidean distance between each pair of points where one point is from `a` and the other is from `b`.
The result is a 2-dimensional array where the element at position (i, j) represents the distance between the i-th point in `a` and the j-th point in `b`.

# Parameters

- `a`: &Array2<f32>
    A two-dimensional array where each row represents a point in n-dimensional space. The dimensions of the array are `[number_of_points_a, dimensions_of_each_point]`.

- `b`: &Array2<f32>
    A two-dimensional array similar to `a`, where each row represents a point in n-dimensional space. The dimensions of the array are `[number_of_points_b, dimensions_of_each_point]`.
    It is not required for `a` and `b` to have the same number of points (rows), but they must be in the same n-dimensional space (have the same number of columns).

# Returns

- `Array2<f32>`
    A two-dimensional array of floating-point numbers, where each element (i, j) is the Euclidean distance between the i-th point in `a` and the j-th point in `b`. The resulting array has dimensions `[number_of_points_a, number_of_points_b]`.

This function is useful for computing distances between two sets of points in machine learning algorithms, such as clustering, where the calculation of distances between points is a common operation.
*/
fn cdist_euclidean(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let mut distances = Array2::<f32>::zeros((a.nrows(), b.nrows()));
    for (i, a_row) in a.outer_iter().enumerate() {
        for (j, b_row) in b.outer_iter().enumerate() {
            let distance = a_row
                .iter()
                .zip(b_row.iter())
                .map(|(&a_val, &b_val)| (a_val - b_val).powi(2))
                .sum::<f32>()
                .sqrt();
            distances[[i, j]] = distance;
        }
    }
    distances
}

/**
Calculates the correlation coefficient matrix for a given dataset.

The correlation coefficient matrix, often denoted as R, is a measure of the strength and direction of a linear relationship between two variables.
This function computes the correlation coefficients for every pair of variables in the input dataset, returning a matrix where each element (i, j) represents the correlation coefficient between the i-th and j-th variables.

# Type Parameters

- `S`: The storage type of the array, which must satisfy the `Data` trait with `Elem = f32`. This allows the function to work with different array storage types while ensuring the elements are floating-point numbers.

# Parameters

- `x`: &ArrayBase<S, Ix2>
    A two-dimensional array where each column represents a variable and each row represents an observation. The input array `x` must have floating-point numbers (`f32`).

# Returns

- `Array2<f32>`
    A two-dimensional array of floating-point numbers representing the correlation coefficient matrix of the input variables. The dimensions of the returned matrix are `(x.ncols(), x.ncols())`,
    where each element (i, j) is the correlation coefficient between the i-th and j-th variables in the input dataset.
*/
fn corrcoef(x: &Array2<f32>) -> Array2<f32> {
    let x = x.t();
    let means = x.mean_axis(Axis(0)).unwrap();
    let mut cov_matrix = Array2::<f32>::zeros((x.ncols(), x.ncols()));

    for i in 0..x.ncols() {
        for j in 0..x.ncols() {
            let xi = x.column(i).to_owned() - means[i];
            let xj = x.column(j).to_owned() - means[j];
            let cov = xi.dot(&xj) / xi.len() as f32;
            cov_matrix[[i, j]] = cov;
        }
    }

    // Convert covariance matrix to correlation coefficient matrix
    let variances = cov_matrix.diag().mapv(|v| v.sqrt());
    for i in 0..cov_matrix.nrows() {
        for j in 0..cov_matrix.ncols() {
            if i != j {
                cov_matrix[[i, j]] /= variances[i] * variances[j];
            }
        }
    }

    // Ensure diagonal elements are 1.0
    for i in 0..cov_matrix.nrows() {
        cov_matrix[[i, i]] = 1.0;
    }

    cov_matrix
}

/**
Finds the maximum absolute value of the off-diagonal elements in a square matrix.

This function calculates the maximum absolute value among all off-diagonal elements of the input matrix `r`.
It is useful for identifying the largest element outside the main diagonal, which can be indicative of the need for further matrix operations or adjustments in numerical methods.

# Parameters

- `r`: &Array2<f32>
    A two-dimensional square array of floating-point numbers. The function considers elements outside the main diagonal for its calculation.

# Returns

- `f32`
    The maximum absolute value among all off-diagonal elements of the matrix `r`.

This function is commonly used in numerical analysis and matrix computations, especially in algorithms that involve matrix diagonalization or convergence checks.
*/
fn max_abs_off_diagonal(r: &Array2<f32>) -> f32 {
    let identity: Array2<f32> = Array2::eye(r.nrows());
    let abs_diff = (r - identity).mapv_into(f32::abs);
    let max_abs_off_diag = abs_diff.iter().fold(0.0_f32, |acc, &x| acc.max(x));

    max_abs_off_diag
}

/**
Calculates the mean of the first two non-NaN values in a slice of floating-point numbers.

This function processes an input slice of floating-point numbers, filtering out any NaN values, and then calculates the average of the first two valid (non-NaN) numbers.
If the slice contains fewer than two non-NaN values, the function calculates the mean of available non-NaN values. If no valid values are found, the function returns 0.0, representing an undefined mean due to the absence of valid inputs.

# Parameters

- `values`: &[f32]
    A slice of floating-point numbers, potentially containing NaN values, from which the mean of the first two valid values will be calculated.

# Returns

- `f32`
    The mean of the first two non-NaN values in the input slice. If the input contains fewer than two non-NaN values, the mean of available non-NaN values is returned. If no valid values are present, 0.0 is returned.

# Note

- This function is designed to ignore NaN values, which can be common in datasets with missing or undefined values. It ensures that calculations are based only on valid numerical data.

This function is useful in data processing and analysis tasks where it is necessary to compute statistics on datasets that may include missing or undefined values.
*/
fn mean_of_first_two(values: &[f32]) -> f32 {
    let valid_values: Vec<f32> = values.iter().filter(|v| !v.is_nan()).cloned().collect();
    let total: f32 = valid_values.iter().take(2).sum();
    total / valid_values.len() as f32
}

/**
Removes specified rows from a 2D array and returns the resulting array.

Given a 2D array and a list of row indices, this function creates a new 2D array with the specified rows removed.
The function is careful to handle indices that fall outside the range of the array's row count by simply ignoring them.
This is particularly useful in data manipulation tasks where certain observations (rows) need to be excluded based on some criteria.

# Parameters

- `arr`: Array2<f32>
    The original two-dimensional array from which rows will be removed. It should be of any size but with floating-point numbers (f32).

- `indices`: Array1<usize>
    An array of row indices to be removed from `arr`. Indices should be zero-based and can be in any order. If an index is out of bounds (i.e., greater than or equal to `arr.nrows()`), it will be ignored.

# Returns

- `Array2<f32>`
    A new two-dimensional array that is a copy of `arr` with the specified rows removed. If all specified indices are out of bounds, the returned array will be identical to `arr`.
*/
fn delete_rows(arr: Array2<f32>, indices: Array1<usize>) -> Array2<f32> {
    let mut to_delete = vec![false; arr.nrows()];
    for &index in indices.iter() {
        if index < to_delete.len() {
            to_delete[index] = true;
        }
    }

    let mut new_vec = Vec::new();
    for (i, row) in arr.axis_iter(Axis(0)).enumerate() {
        if !to_delete[i] {
            new_vec.push(row.to_owned());
        }
    }

    match stack(
        Axis(0),
        &new_vec.iter().map(|a| a.view()).collect::<Vec<_>>(),
    ) {
        Ok(res) => res,
        Err(_) => panic!("Error stacking rows back into an array."),
    }
}

/**
Sorts each row of a 2D array, handling NaN values by placing them at the end of each row.

This function iterates over each row of a 2D array and sorts the elements in ascending order, with the exception that NaN values are treated as greater than any number.
This ensures that NaN values are moved to the end of each row after sorting. The sorting is stable for non-NaN values, preserving their relative order when possible.

# Parameters

- `array`: Array2<f32>
    A mutable reference to a two-dimensional array of floating-point numbers. The array is modified in place, with each row sorted according to the rules specified.

# Returns

- `Array2<f32>`
    The same array passed in, with each row sorted such that numerical values are in ascending order and NaN values are placed at the end of each row. This allows for easier handling of NaN values in subsequent data processing steps.

The function's approach to handling NaN values makes it particularly useful in data processing and analysis tasks where NaN represents missing or undefined data that should not interfere with sorting operations.
*/
fn sort_array2_by_axis_with_nan_handling(mut array: Array2<f32>) -> Array2<f32> {
    for mut row in array.axis_iter_mut(Axis(0)) {
        let mut row_vec: Vec<f32> = row.to_vec();

        // Custom sort to handle NaN values. We use partial_cmp for comparison
        // and specify that NaN values should be considered as greater
        row_vec.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
        });

        // Update the original row with sorted values
        row.assign(
            &Array2::from_shape_vec((1, row_vec.len()), row_vec)
                .unwrap()
                .row(0),
        );
    }

    array
}

/**
Finds the index of the minimum non-NaN value in a slice of floating-point numbers.

This function iterates over a slice of `f32` values, ignoring any NaN values, and returns the index of the minimum value found. If the slice contains only NaN values or is empty, it returns `None`.
This is useful for data analysis tasks where NaN values represent missing data and should not be considered in minimum value calculations.

# Parameters

- `vec`: &[f32]
    A slice of floating-point numbers which may include NaN values alongside regular floating-point numbers.

# Returns

- `Option<usize>`
    An `Option` containing the index of the minimum non-NaN value in the slice. Returns `None` if the slice is empty or contains only NaN values, indicating that a minimum value could not be determined under the given conditions.

This function is particularly useful in statistical computations and data preprocessing, where it's common to encounter and need to gracefully handle NaN values.

 */
fn argmin_ignore_nan(vec: &[f32]) -> Option<usize> {
    vec.iter()
        .enumerate()
        .filter_map(|(index, &value)| {
            if value.is_nan() {
                None
            } else {
                Some((index, value))
            }
        })
        .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
}

/**
Sorts each column of a 2D array and returns an array of sorted indices.

Given a 2D array of floating-point numbers, this function sorts the values in each column while handling NaN values by placing them at the end of the sorting order.
Instead of sorting the array itself, it returns a new 2D array where each element is the original index of the corresponding sorted element in the input array.
This is useful for tasks that require sorting data while retaining a mapping back to the original data order.

# Parameters

- `array`: &Array2<f32>
    A reference to a two-dimensional array of floating-point numbers. The array is not modified by this function.

# Returns

- `Array2<usize>`
    A two-dimensional array of the same shape as `array`, where each element in the array represents the original index of the corresponding sorted element in each column of the input array.

This function is particularly useful in data analysis and preprocessing where sorting is needed but the original indices must be retained for further processing or analysis.

*/
fn argsort_axis0(array: &Array2<f32>) -> Array2<usize> {
    let mut sorted_indices = Array2::default((array.nrows(), array.ncols()));

    for (j, column) in array.axis_iter(Axis(1)).enumerate() {
        // Collect indices and values into a vector
        let mut pairs: Vec<(usize, &f32)> = column.iter().enumerate().collect();

        // Sort by the values, handling NaN values by pushing them to the end
        pairs.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Greater));

        // Extract the sorted indices and assign them to the corresponding column in the result
        for (i, (sorted_index, _)) in pairs.iter().enumerate() {
            sorted_indices[[i, j]] = *sorted_index;
        }
    }

    sorted_indices
}

// ##############################################################################################################
// -------------------------------------------------- Tests -----------------------------------------------------
// ##############################################################################################################

#[cfg(test)]
#[allow(dead_code)]
mod tests {

    // Import the outer module to use the function to be tested.
    use super::*;
    use ndarray::{Array3, ArrayBase, Data, Dimension};

    // ######################################### helper functions ######################################
    fn sort_ndarray_array1(array: Array1<f32>) -> Array1<f32> {
        let mut vec = array.to_vec();
        vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Array1::from(vec)
    }

    fn arrays_are_close<S, D>(a: &ArrayBase<S, D>, b: &ArrayBase<S, D>, tolerance: f32) -> bool
    // checks if all the Array1 elements are within tolerance
    where
        S: Data<Elem = f32>,
        D: Dimension,
    {
        assert_eq!(a.shape(), b.shape(), "array shapes must be the same");
        Zip::from(a)
            .and(b)
            .fold(true, |acc, &a, &b| acc && (a - b).abs() <= tolerance)
    }

    fn vec_array2_to_array3(arrays: Vec<Array2<f32>>) -> Array3<f32> {
        let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
        stack(Axis(0), &views).expect("Error stacking arrays")
    }

    fn test_average_value(vectors: Vec<Array2<f32>>, tolerance: f32) -> bool {
        // takes a vector of Array2<f32>
        let collection_array = vec_array2_to_array3(vectors);

        // get the average Array2<f32>
        let avg_array = collection_array.mean_axis(Axis(0)).unwrap();

        let shape_arr = avg_array.shape();

        let samples = shape_arr[0];
        let n = shape_arr[1];

        // make Array2 of the same size with 0.5 elements
        let center = Array2::from_elem((samples, n), 0.5);

        // make sure that the Array2s are close to eachother
        arrays_are_close(&center, &avg_array.to_owned(), tolerance)
    }

    // ######################################### tests #################################################

    mod test_averages {
        use super::*;

        // makes sure that after enough iteration, all the values in the arrays settle close to 0.5 as an average
        // so that the fuctions arent biased in the long run

        #[test]
        fn lhs_classic_average() {
            let n = 13;
            let samples = 12;

            let mut vectors: Vec<Array2<f32>> = vec![];
            for i in 0..1000 {
                vectors.push(lhs_classic(n, samples, i));
            }

            let tolerance = 0.05;
            assert!(test_average_value(vectors, tolerance));
        }

        #[test]
        fn lhs_centered_average() {
            let n = 2;
            let samples = 3;

            let mut vectors: Vec<Array2<f32>> = vec![];
            for i in 0..1000 {
                vectors.push(lhs_centered(n, samples, i));
            }

            let tolerance = 0.05;
            assert!(test_average_value(vectors, tolerance));
        }

        #[test]
        fn lhs_maximin_average() {
            let n = 10;
            let samples = 15;
            let iterations = 4;

            let mut vectors: Vec<Array2<f32>> = vec![];
            for i in 0..1000 {
                vectors.push(lhs_maximin(n, samples, i, iterations));
            }

            let tolerance = 0.05;
            assert!(test_average_value(vectors, tolerance));
        }

        #[test]
        fn lhs_correlate_average() {
            let n = 14;
            let samples = 11;
            let iterations = 5;

            let mut vectors: Vec<Array2<f32>> = vec![];
            for i in 0..1000 {
                vectors.push(lhs_correlate(n, samples, i, iterations));
            }

            let tolerance = 0.05;
            assert!(test_average_value(vectors, tolerance));
        }

        #[test]
        fn lhs_mu_average() {
            let n = 5;
            let samples = 6;

            let mut vectors: Vec<Array2<f32>> = vec![];
            for i in 0..1000 {
                vectors.push(lhs_mu(n, samples, i));
            }

            let tolerance = 0.1;
            assert!(test_average_value(vectors, tolerance));
        }
    }

    mod test_guarantees {
        use super::*;

        // makes sure that some guarantees that the functions offer are indeed kept

        #[test]
        fn lhs_centered_guarantee() {
            let n = 6;
            let samples = 20;
            let random_state = 42;

            let arr = lhs_centered(n, samples, random_state);

            let mut cut: Array1<f64> = Array::linspace(0., 1., samples + 1);
            cut = cut.mapv(|x: f64| (x * 100.).round() / 100.); // rounding the variables to 0.01

            let a = cut.slice(s![..samples]);
            let b = cut.slice(s![1..samples + 1]);
            let center = ((&a + &b) / 2.).mapv(|x| x as f32);
            let sorted_center = sort_ndarray_array1(center);

            for col in arr.axis_iter(Axis(1)) {
                let sorted_col = sort_ndarray_array1(col.to_owned());
                assert_eq!(sorted_center, sorted_col);
            }
        }

        #[test]
        fn lhs_mu_guarantee() {
            let n = 17;
            let samples = 15;
            let random_state = 42;
            let arr = lhs_mu(n, samples, random_state);

            let mut cut: Array1<f64> = Array::linspace(0., 1., samples + 1);
            cut = cut.mapv(|x: f64| (x * 100.).round() / 100.); // rounding the variables to 0.01

            let a = cut.slice(s![..samples]);
            let b = cut.slice(s![1..samples + 1]);
            let mut center = ((&a + &b) / 2.).mapv(|x| x as f32);
            center = sort_ndarray_array1(center);

            for col in arr.axis_iter(Axis(1)) {
                let sorted_col = sort_ndarray_array1(col.to_owned());

                assert!(arrays_are_close(
                    &center,
                    &sorted_col,
                    0.5 / (samples as f32)
                ));
            }
        }
    }

    mod utilities_tests {

        use ndarray::{array, Array2};

        use crate::random_design::{corrcoef, pairwise_euclidean_dist, tests::arrays_are_close};

        #[test]
        fn euclidean_distance_test() {
            // making sure that the re-implementation of scipy.spatial.distance.pdist is correct
            let test_arr = array![
                [-1., -2., -3.],
                [1., 2., 3.],
                [10., -15., 32.],
                [-100., 340., 32.],
                [-342., 421., -523.],
            ];
            let expected = array![
                7.483315, 38.923, 357.7569, 752.0705, 34.799427, 353.9576, 754.90796, 371.65173,
                788.6856, 610.86005,
            ];
            let result_array = pairwise_euclidean_dist(&test_arr);
            assert!(arrays_are_close(&result_array, &expected, 0.01));
        }

        #[test]
        fn correlation_test() {
            // making sure that the re-implementation of np.corrcoef is correct
            let test_arr: Array2<f32> = array![
                [-1., -2., -3.],
                [1., 2., 3.],
                [10., -15., 32.],
                [-100., 340., 32.],
                [-342., 421., -523.],
            ];
            let expected = array![
                [1.0, -0.89002377, 0.9418991],
                [-0.89002377, 1.0, -0.6858651],
                [0.9418991, -0.6858651, 1.0],
            ];

            let return_array = corrcoef(&test_arr.t().to_owned());

            assert_eq!(
                return_array.shape(),
                expected.shape(),
                "array shapes do not match, {:?} vs expected: {:?}",
                return_array.shape(),
                expected.shape()
            );

            assert!(arrays_are_close(&return_array, &expected, 0.01));
        }
    }
}
