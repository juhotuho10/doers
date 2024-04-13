use itertools::Itertools;
use ndarray::{concatenate, s, Array, Array1, Array2, Array3, ArrayViewMut1, Axis};
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
Creates a general full-factorial design.

# Parameters

- `levels`: Vec<u16>
  A vector indicating the number of levels for each input design factor.
  Each element in the vector represents a different factor and specifies how many levels that factor has.

# Returns

- Result<Array2<u16>, String>
  Returns a design matrix with coded levels ranging from 0 to `k-1` for a `k`-level factor.
  The matrix represents all possible combinations of the levels across all factors.

# Errors

- Returns an error string if any of the following conditions are met:
  - Any value in the `levels` vector is `0`, as a factor cannot have zero levels.
  - The cumulative product of the `levels` values exceeds `usize::MAX`, indicating the resulting matrix would be too large to handle.

# Example

Generate a full-factorial design for three factors with 2, 4, and 3 levels respectively:

```rust
use doers::factorial_design::fullfact;
let example_array = fullfact(vec![2, 4, 3]);
//
// resulting Array2<u16>:
//
// [[ 0,  0,  0],
//  [ 1,  0,  0],
//  [ 0,  1,  0],
//  [ 1,  1,  0],
//  [ 0,  2,  0],
//  [ 1,  2,  0],
//  [ 0,  3,  0],
//  [ 1,  3,  0],
//  [ 0,  0,  1],
//  [ 1,  0,  1],
//  [ 0,  1,  1],
//  [ 1,  1,  1],
//  [ 0,  2,  1],
//  [ 1,  2,  1],
//  [ 0,  3,  1],
//  [ 1,  3,  1],
//  [ 0,  0,  2],
//  [ 1,  0,  2],
//  [ 0,  1,  2],
//  [ 1,  1,  2],
//  [ 0,  2,  2],
//  [ 1,  2,  2],
//  [ 0,  3,  2],
//  [ 1,  3,  2]];
```
 */

pub fn fullfact(levels: Vec<u16>) -> Result<Array2<u16>, String> {
    let n = levels.len();
    let num_lines: u64 = levels.iter().map(|&level| level as u64).product();

    if num_lines > usize::MAX as u64 {
        return Err("Number of lines exceeds maximum allowable size.".to_string());
    } else if levels.iter().any(|x| *x == 0) {
        return Err("All level sizes must be 1 or higher".to_string());
    }

    let mut array: Array2<u16> = Array2::<u16>::zeros((num_lines as usize, n));
    let mut level_repeat = 1;
    let mut range_repeat = num_lines as usize;

    for (i, &level) in levels.iter().enumerate() {
        range_repeat /= level as usize;

        // taking the slice of the array and mutating it in place without copying or moving
        let slice = array.slice_mut(s![.., i]);
        build_level(slice, level, level_repeat, range_repeat);
        level_repeat *= level as usize;
    }
    Ok(array)
}

fn build_level(mut output: ArrayViewMut1<u16>, level: u16, repeat: usize, range_repeat: usize) {
    let mut count = 0;
    for _ in 0..range_repeat {
        for j in 0..level {
            for _ in 0..repeat {
                output[count] = j;
                count += 1;
            }
        }
    }
}

/**
Creates a 2-Level full-factorial design.

# Parameters

- `n`: usize
  The number of factors in the design. This determines the size and complexity of the resulting matrix.

# Returns

- `Array2<i32>`
  The design matrix with coded levels -1 and 1, representing the two levels for each factor across all possible combinations.

# Errors

- Raises a `ValueError` if `n` is too large (in the thousands) and 2^n causes u64 to overflow

# Example

Generate a full-factorial design for 3 factors:

```rust
use doers::factorial_design::ff2n;
let example_array = ff2n(3);
//
// resulting Array2<i32>:
//
// array([[-1., -1., -1.],
//        [ 1., -1., -1.],
//        [-1.,  1., -1.],
//        [ 1.,  1., -1.],
//        [-1., -1.,  1.],
//        [ 1., -1.,  1.],
//        [-1.,  1.,  1.],
//        [ 1.,  1.,  1.]])
```
*/
#[allow(dead_code)]
pub fn ff2n(n: usize) -> Result<Array2<i32>, String> {
    let vec: Vec<u16> = vec![2; n];

    let return_vec = fullfact(vec);

    match return_vec {
        Ok(return_vec) => {
            let return_vec = return_vec.mapv(|x| x as i32);
            Ok(2 * return_vec - 1)
        }
        Err(return_vec) => Err(return_vec),
    }
}

/**
Generates a Plackett-Burman design.

# Parameter

- `n`: u32
  The number of factors for which to create a matrix.

# Returns

- Array2<i32>
  Returns an orthogonal design matrix with `n` columns, one for each factor. The number of rows is the next multiple of 4 higher than `n`. For example, for 1-3 factors, there are 4 rows; for 4-7 factors, there are 8 rows, etc.

# Errors

- Raises a `ValueError` if:
  - The input is valid, or if the design construction fails. The design can fail if `reduction` is too large compared to the values of `levels`. Note: It seems there might be a typo in the original text regarding `reduction` and `levels` since they are not parameters of this function. It's possible this section was mistakenly included from another function's documentation.

# Examples

Generate a design for 5 factors:
```
use doers::factorial_design::pbdesign;
let example_array = pbdesign(5);
//
// resulting Array2<i32>:
//
// array([[-1, -1,  1, -1,  1],
//        [ 1, -1, -1, -1, -1],
//        [-1,  1, -1, -1,  1],
//        [ 1,  1,  1, -1, -1],
//        [-1, -1,  1,  1, -1],
//        [ 1, -1, -1,  1,  1],
//        [-1,  1, -1,  1, -1],
//        [ 1,  1,  1,  1,  1]])
```
*/

#[allow(dead_code)]
pub fn pbdesign(n: u32) -> Array2<i32> {
    let keep = n as usize;
    let n = n as f32;

    let n = 4. * ((n / 4.).floor() + 1.); // calculate the correct number of rows

    let mut k = 0;
    let num_array = Array::from_vec(vec![n, n / 12., n / 20.]);

    let (significand, exponents) = frexp(&num_array);

    for (index, (&mantissa, &exponent)) in significand.iter().zip(exponents.iter()).enumerate() {
        if mantissa == 0.5 && exponent > 0 {
            k = index;
            break;
        }
    }
    let exp = exponents[k] - 1;
    let mut h: Array2<i32>;

    match k {
        0 => {
            h = Array2::ones((1, 1));
        }
        1 => {
            let top: Array2<i32> = Array2::ones((1, 12));
            let bottom_left: Array2<i32> = Array2::ones((11, 1));
            let bottom_right: Array2<i32> = toeplitz(
                &[-1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1],
                &[-1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],
            );
            let bottom: Array2<i32> = concatenate![Axis(1), bottom_left, bottom_right];
            h = concatenate![Axis(0), top, bottom];
        }
        2 => {
            let top: Array2<i32> = Array2::ones((1, 20));
            let bottom_left: Array2<i32> = Array2::ones((19, 1));
            let bottom_right: Array2<i32> = hankel(
                &[
                    -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1,
                ],
                &[
                    1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1,
                ],
            );
            let bottom: Array2<i32> = concatenate![Axis(1), bottom_left, bottom_right];
            h = concatenate![Axis(0), top, bottom];
        }
        _ => unreachable!("Invalid value for k, this shouldn't happen"),
    };

    for _ in 0..exp {
        let h_top: Array2<i32> = concatenate![Axis(1), h.clone(), h.clone()];
        let h_bottom: Array2<i32> = concatenate![Axis(1), h.clone(), -h.clone()];
        h = concatenate![Axis(0), h_top, h_bottom];
    }

    // Reduce the size of the matrix as needed
    let keep = keep.min(h.shape()[0]);
    let h: Array2<i32> = h.slice(s![.., 1..=keep]).to_owned();

    // Flip the matrix upside down
    let h_flipped: Array2<i32> = h.slice(s![..;-1, ..]).to_owned();
    h_flipped
}

/**
 Creates a Generalized Subset Design (GSD).

 # Parameters

 - `levels`: array-like
   Number of factor levels per factor in design.

 - `reduction`: int
   Reduction factor (greater than 1). A larger `reduction` means fewer
   experiments in the design and more possible complementary designs.

 - `n`: int
   Number of complementary GSD-designs (default is 1). The complementary
   designs are balanced analogous to fold-over in two-level fractional
   factorial designs.

 # Returns

 - `Vec<Array2<u16>>` n amount of complementary Array2<u16> matrices that have complementary designs,
    where the design size will be reduced down by reduction size

 # Raises

 - `ValueError`
   If input is valid or if design construction fails. The design can fail
   if `reduction` is too large compared to values of `levels`.

 # Notes

 The Generalized Subset Design (GSD) or generalized factorial design is
 a generalization of traditional fractional factorial designs to problems
 where factors can have more than two levels.

 In many application problems, factors can have categorical or quantitative
 factors on more than two levels. Previous reduced designs have not been
 able to deal with such types of problems. Full multi-level factorial
 designs can handle such problems but are, however, not economical regarding
 the number of experiments.

 Note for commercial users, the application of GSD to testing of product
 characteristics in a processing facility is patented.

 # Examples

 A single design of 3, 4 and 4 levels with a reduction of 2 and

 ```
use doers::factorial_design::gsd;
let levels = vec![3,4,4];
let reductions = 2;
let n_arrays = 1;
let example_array = gsd(levels, reductions, n_arrays);
//
// resulting Vec<Array2<u16>>:
// [[[0, 0, 0],
//  [0, 0, 2],
//  [0, 2, 0],
//  [0, 2, 2],
//  [2, 0, 0],
//  [2, 0, 2],
//  [2, 2, 0],
//  [2, 2, 2],
//  [0, 1, 1],
//  [0, 1, 3],
//  [0, 3, 1],
//  [0, 3, 3],
//  [2, 1, 1],
//  [2, 1, 3],
//  [2, 3, 1],
//  [2, 3, 3],
//  [1, 0, 1],
//  [1, 0, 3],
//  [1, 2, 1],
//  [1, 2, 3],
//  [1, 1, 0],
//  [1, 1, 2],
//  [1, 3, 0],
//  [1, 3, 2]]
 ```

 Example of Two complementary designs:

 ```
use doers::factorial_design::gsd;
let levels = vec![3,3];
let reductions = 2;
let n_arrays = 2;
let example_array = gsd(levels, reductions, n_arrays);
//
// resulting Vec<Array2<u16>>:
// [[[0, 0],
//  [0, 2],
//  [2, 0],
//  [2, 2],
//  [1, 1]],
//  [[0, 1],
//  [2, 1],
//  [1, 0],
//  [1, 2]]]
 ```

 If the design fails a ValueError is raised:

 ```
use doers::factorial_design::gsd;
let levels = vec![2,3];
let reductions = 5;
let n_arrays = 1;
let example_array = gsd(levels, reductions, n_arrays);
 // Returns an Err: Err("Reduction is too large for the design size")
 ```

 # References

 - Surowiec, Izabella, et al. "Generalized Subset Designs in Analytical Chemistry." Analytical Chemistry 89.12 (2017): 6491-6497. <https://doi.org/10.1021/acs.analchem.7b00506>
 - Vikstrom, Ludvig, et al. Computer-implemented systems and methods for generating generalized fractional designs. US9746850 B2, filed May 9, 2014, and issued August 29, 2017. <http://www.google.se/patents/US9746850>
*/
#[allow(dead_code)]
pub fn gsd(levels: Vec<u16>, reduction: usize, n: usize) -> Result<Vec<Array2<u16>>, String> {
    assert!(reduction > 1, "The level of reductions must 2 or higher");
    if reduction < 2 {
        return Err("The level of reductions must 2 or higher".to_string());
    } else if n < 1 {
        return Err("n number of designs must be 1 or higher".to_string());
    }

    let partitions: Vec<Vec<Vec<u16>>> = make_partitions(&levels, &reduction);
    let latin_square: Array2<u16> = make_latin_square(reduction);
    let ortogonal_arrays: Array3<u16> = make_orthogonal_arrays(&latin_square, levels.len());

    let mut design_vec: Vec<Array2<u16>> = vec![];

    for oa in ortogonal_arrays.axis_iter(Axis(0)) {
        // call the function with orthagonal arrays and forward the error if the function fails
        let design: Array2<u16> = map_partitions_to_design(&partitions, &oa.to_owned())?;
        design_vec.push(design);
    }

    Ok(design_vec.iter().take(n).cloned().collect())
}

fn make_partitions(factor_levels: &[u16], &num_partitions: &usize) -> Vec<Vec<Vec<u16>>> {
    // Calculate total partitions and maximum size to initialize the array.
    //let max_size = *factor_levels.iter().max().unwrap_or(&1) as usize;
    let mut partitions_vec: Vec<Vec<Vec<u16>>> = vec![];

    for partition_idx in 0..num_partitions {
        let mut partition: Vec<Vec<u16>> = vec![];
        for &num_levels in factor_levels {
            let mut part: Vec<u16> = Vec::new();
            for level_i in 1..num_levels {
                let index = (partition_idx + 1) as u16 + (level_i - 1) * num_partitions as u16;
                if index <= num_levels {
                    part.push(index);
                }
            }
            partition.push(part);
        }
        partitions_vec.push(partition);
    }
    partitions_vec
}

fn make_latin_square(n: usize) -> Array2<u16> {
    let return_array: Array2<u16> = Array2::from_shape_fn((n, n), |(row, col)| {
        let roll = (col + row) % n;
        roll as u16
    });
    return_array
}
fn make_orthogonal_arrays(latin_square: &Array2<u16>, n_cols: usize) -> Array3<u16> {
    let first_row = latin_square.slice(s![0, ..]);

    let mut a_matrices: Vec<Array2<u16>> = first_row
        .iter()
        .map(|&v| Array::from_elem((1, 1), v))
        .collect();

    while a_matrices[0].shape()[1] < n_cols {
        let mut new_a_matrices = vec![];

        for i in 0..a_matrices.len() {
            let mut sub_a: Vec<Array2<u16>> = vec![];

            let indexes = latin_square.slice(s![i, ..]).mapv(|x| x as usize);
            let zip_array: Vec<Array2<u16>> = indexes
                .iter()
                .filter_map(|&i| a_matrices.get(i).cloned())
                .collect();
            for (constant, other_a) in first_row.iter().zip(zip_array.iter()) {
                let repeat_array: Array1<u16> = Array::from_elem(other_a.shape()[0], *constant);
                let repeat_array: Array2<u16> = repeat_array.insert_axis(Axis(1));
                let combined = concatenate![Axis(1), repeat_array, *other_a];
                sub_a.push(combined);
            }
            let new_a_matrix =
                concatenate(Axis(0), &sub_a.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();
            new_a_matrices.push(new_a_matrix);
        }

        a_matrices = new_a_matrices;

        if a_matrices[0].shape()[1] == n_cols {
            break;
        }
    }

    // converting Vec<Array2> into Array3
    let depth = a_matrices.len();
    let (rows, cols) = a_matrices[0].dim();
    let mut return_matrix: Array3<u16> = Array3::zeros((depth, rows, cols));

    for (i, array2) in a_matrices.iter().enumerate() {
        let mut layer = return_matrix.slice_mut(s![i, .., ..]);
        layer.assign(array2);
    }

    return_matrix
}

fn map_partitions_to_design(
    partitions: &[Vec<Vec<u16>>],
    ortogonal_array: &Array2<u16>,
) -> Result<Array2<u16>, String> {
    if !(partitions.len() == *ortogonal_array.iter().max().unwrap() as usize + 1
        && *ortogonal_array.iter().min().unwrap() == 0)
    {
        return Err("Orthogonal array indexing does not match partition structure".to_string());
    }

    let mut mappings: Vec<Vec<u16>> = Vec::new();

    for row in ortogonal_array.axis_iter(Axis(0)) {
        // partitions is Array3, we take the Array1 from position partition[[p, factor]]
        let partition_sets: Vec<Vec<u16>> = row
            .iter()
            .enumerate()
            .map(|(factor, &p)| partitions[p as usize][factor].clone())
            .collect();

        if partition_sets.iter().any(|set| set.iter().all(|&x| x == 0)) {
            continue;
        }

        // Computing the cartesian product of the partition sets
        let cartesian_product = partition_sets.into_iter().multi_cartesian_product();

        // Adding the cartesian products to the mappings
        for product in cartesian_product {
            mappings.push(product);
        }
    }

    if mappings.is_empty() {
        return Err("Reduction is too large for the design size".to_string());
    }

    // Convert mappings to Array2<u16>. You might need to adjust this part based on your specific requirements
    let ncols = mappings[0].len();
    let flat: Vec<u16> = mappings.into_iter().flatten().collect();
    let nrows = flat.len() / ncols;

    Ok(Array2::from_shape_vec((nrows, ncols), flat).unwrap() - 1)
}

// ##############################################################################################################
// -------------------------------------------------- Utils -----------------------------------------------------
// ##############################################################################################################

/**
frexp for a Array1 of f32
Decomposes elements of a float array into mantissas and exponents.

Each float `x` is transformed into `m * 2^e`, where `m` is the mantissa and `e` the exponent.

# Arguments
`Array1<f32>`

# Returns
A tuple of arrays (`Array1<f32>`, `Array1<i32>`) for mantissas and exponents respectively.
*/

fn frexp(arr: &Array1<f32>) -> (Array1<f32>, Array1<i32>) {
    let mantissas = arr.mapv(|x| {
        if x == 0.0 {
            0.0 // Mantissa for zero
        } else {
            let mut y = x.abs();
            let exponent = y.log2().floor() + 1.0;
            y /= f32::powf(2.0, exponent);
            if x < 0.0 {
                -y
            } else {
                y
            }
        }
    });

    let exponents = arr.mapv(|x| {
        if x == 0.0 {
            0 // Exponent for zero
        } else {
            (x.abs().log2().floor() + 1.0) as i32
        }
    });

    (mantissas, exponents)
}

/**
Reimplemented Toeplitz function from Python scipy.linalg
Generates a Toeplitz matrix given the first column and first row values.

A Toeplitz matrix is characterized by constant diagonals, with each descending diagonal
from left to right being constant.

# Arguments
* `c` - A slice of `i32`, representing the first column of the matrix.
* `r` - A slice of `i32`, representing the first row of the matrix.

# Returns
Returns an `Array2<i32>` where each element at position (i, j) is determined by `c[i - j]`
*/
fn toeplitz(c: &[i32], r: &[i32]) -> Array2<i32> {
    let c_len = c.len();
    let r_len = r.len();

    // Ensure the first element of r is ignored in the output matrix as per Python's scipy implementation
    let mut new_vec = Array2::<i32>::zeros((c_len, r_len));

    for (i, mut row) in new_vec.rows_mut().into_iter().enumerate() {
        for (j, elem) in row.iter_mut().enumerate() {
            *elem = if i > j { c[i - j] } else { r[j - i] };
        }
    }

    new_vec
}

/**
Reimplemented Hankel function from Python scipy.linalg
Constructs a Hankel matrix from initial column and row vectors.

# Arguments
* `c` - Initial column as a slice of `i32`, setting the first column of the matrix.
* `r` - Last row as a slice of `i32`, defining the continuation from the end of `c`.

# Returns
Returns an `Array2<i32>` representing the Hankel matrix.
*/
fn hankel(c: &[i32], r: &[i32]) -> Array2<i32> {
    let c = Array1::from_vec(c.to_vec());
    let r = Array1::from_vec(r.to_vec());

    let n = c.len();
    let m = r.len();

    // Create an uninitialized 2D array
    let mut matrix = Array2::<i32>::zeros((n, m));

    // Fill the Hankel matrix
    for i in 0..n {
        for j in 0..m {
            if i + j < n {
                matrix[[i, j]] = c[i + j];
            } else {
                matrix[[i, j]] = r[(i + j) % n + 1];
            }
        }
    }

    matrix
}

// ##############################################################################################################
// -------------------------------------------------- Tests -----------------------------------------------------
// ##############################################################################################################

#[cfg(test)]
mod tests {
    // Testing the functions with a known output
    use super::*;
    use ndarray::Zip;
    fn array1_are_close(a: &Array1<f32>, b: &Array1<f32>, tolerance: f32) -> bool {
        // checks if all the Array1 elements are within tolerance
        Zip::from(a)
            .and(b)
            .fold(true, |acc, &a, &b| acc && (a - b).abs() <= tolerance)
    }

    mod functionality_tests {
        use ndarray::arr2;

        use super::*;

        #[test]
        fn fullfact_1() {
            let input = vec![];
            let expected = arr2(&[[]]);
            assert_eq!(fullfact(input).unwrap(), expected);
        }

        #[test]
        fn fullfact_2() {
            let input = vec![1, 2];
            let expected = arr2(&[[0, 0], [0, 1]]);
            assert_eq!(fullfact(input).unwrap(), expected);
        }

        #[test]
        fn fullfact_3() {
            let input = vec![1, 2, 3];
            let expected = arr2(&[
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 1, 1],
                [0, 0, 2],
                [0, 1, 2],
            ]);
            assert_eq!(fullfact(input).unwrap(), expected);
        }

        #[test]
        fn ff2n_1() {
            let input = 1;
            let expected = arr2(&[[-1], [1]]);
            assert_eq!(ff2n(input).unwrap(), expected);
        }

        #[test]
        fn ff2n_2() {
            let input = 2;
            let expected = arr2(&[[-1, -1], [1, -1], [-1, 1], [1, 1]]);
            assert_eq!(ff2n(input).unwrap(), expected);
        }

        #[test]
        fn ff2n_3() {
            let input = 4;
            let expected = arr2(&[
                [-1, -1, -1, -1],
                [1, -1, -1, -1],
                [-1, 1, -1, -1],
                [1, 1, -1, -1],
                [-1, -1, 1, -1],
                [1, -1, 1, -1],
                [-1, 1, 1, -1],
                [1, 1, 1, -1],
                [-1, -1, -1, 1],
                [1, -1, -1, 1],
                [-1, 1, -1, 1],
                [1, 1, -1, 1],
                [-1, -1, 1, 1],
                [1, -1, 1, 1],
                [-1, 1, 1, 1],
                [1, 1, 1, 1],
            ]);
            assert_eq!(ff2n(input).unwrap(), expected);
        }

        #[test]
        fn pbdesign_1() {
            // uses k=0
            let input = 2;
            let expected = arr2(&[[-1, -1], [1, -1], [-1, 1], [1, 1]]);
            assert_eq!(pbdesign(input), expected);
        }

        #[test]
        fn pbdesign_2() {
            // uses k=0
            let input = 4;
            let expected = arr2(&[
                [-1, -1, 1, -1],
                [1, -1, -1, -1],
                [-1, 1, -1, -1],
                [1, 1, 1, -1],
                [-1, -1, 1, 1],
                [1, -1, -1, 1],
                [-1, 1, -1, 1],
                [1, 1, 1, 1],
            ]);
            assert_eq!(pbdesign(input), expected);
        }

        #[test]
        fn pbdesign_3() {
            // uses k=1
            let input = 8;
            let expected = arr2(&[
                [1, -1, 1, 1, 1, -1, -1, -1],
                [-1, 1, 1, 1, -1, -1, -1, 1],
                [1, 1, 1, -1, -1, -1, 1, -1],
                [1, 1, -1, -1, -1, 1, -1, -1],
                [1, -1, -1, -1, 1, -1, -1, 1],
                [-1, -1, -1, 1, -1, -1, 1, -1],
                [-1, -1, 1, -1, -1, 1, -1, 1],
                [-1, 1, -1, -1, 1, -1, 1, 1],
                [1, -1, -1, 1, -1, 1, 1, 1],
                [-1, -1, 1, -1, 1, 1, 1, -1],
                [-1, 1, -1, 1, 1, 1, -1, -1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]);
            assert_eq!(pbdesign(input), expected);
        }

        #[test]
        fn pbdesign_4() {
            // uses k=2
            let input = 16;
            let expected = arr2(&[
                [1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1],
                [-1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1],
                [-1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1],
                [1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1],
                [1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1],
                [1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1],
                [1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1],
                [-1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1],
                [1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1],
                [-1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1],
                [1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1],
                [-1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1],
                [-1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1],
                [-1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1],
                [-1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1],
                [1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1],
                [1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1],
                [-1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1],
                [-1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]);
            assert_eq!(pbdesign(input), expected);
        }

        #[test]
        fn gsd_1() {
            let levels = vec![2, 2, 3];
            let reductions = 2;
            let n = 1;
            let expected = vec![arr2(&[
                [0, 0, 0],
                [0, 0, 2],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 2],
            ])];
            assert_eq!(gsd(levels, reductions, n), Ok(expected));
        }

        #[test]
        fn gsd_2() {
            let levels = vec![3, 3, 2];
            let reductions = 4;
            let n = 2;
            let expected = vec![
                arr2(&[[0, 0, 0], [0, 1, 1], [1, 0, 1], [2, 2, 0]]),
                arr2(&[[0, 0, 1], [1, 2, 0], [2, 1, 0], [2, 2, 1]]),
            ];
            assert_eq!(gsd(levels, reductions, n), Ok(expected));
        }

        #[test]
        fn gsd_3() {
            let levels = vec![3, 3, 3];
            let reductions = 3;
            let n = 2;
            let expected = vec![
                arr2(&[
                    [0, 0, 0],
                    [0, 1, 1],
                    [0, 2, 2],
                    [1, 0, 1],
                    [1, 1, 2],
                    [1, 2, 0],
                    [2, 0, 2],
                    [2, 1, 0],
                    [2, 2, 1],
                ]),
                arr2(&[
                    [0, 0, 1],
                    [0, 1, 2],
                    [0, 2, 0],
                    [1, 0, 2],
                    [1, 1, 0],
                    [1, 2, 1],
                    [2, 0, 0],
                    [2, 1, 1],
                    [2, 2, 2],
                ]),
            ];
            assert_eq!(gsd(levels, reductions, n), Ok(expected));
        }
    }

    mod utilities_tests {
        use ndarray::{arr1, arr2, Array1};

        use crate::factorial_design::{frexp, hankel, tests::array1_are_close, toeplitz};

        #[test]
        fn frexp_test() {
            let levels: Array1<f32> = arr1(&[5., 5. / 12., 5. / 20.]);
            let (f, e) = frexp(&levels);

            let e = e.mapv(|x| x as f32);

            let expected_f = arr1(&[0.625, 0.8333333, 0.5]);
            let expected_e = arr1(&[3., -1., -1.]);
            assert!(array1_are_close(&f, &expected_f, 0.01));
            assert!(array1_are_close(&e, &expected_e, 0.01));
        }

        #[test]
        fn toeplitz_test() {
            let input_1 = [-1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1];
            let input_2 = [-1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1];
            let expected = arr2(&[
                [-1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],
                [-1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1],
                [1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1],
                [-1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1],
                [-1, -1, 1, -1, -1, 1, -1, 1, 1, 1, -1],
                [-1, -1, -1, 1, -1, -1, 1, -1, 1, 1, 1],
                [1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1],
                [1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1],
                [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
                [-1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1],
                [1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1],
            ]);

            let result = toeplitz(&input_1, &input_2);

            assert_eq!(result, expected);
        }

        #[test]
        fn hankel_test() {
            let input_1 = [-1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1];
            let input_2 = [1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1];
            let expected = arr2(&[
                [-1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1],
                [-1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1],
                [1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1],
                [1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1],
                [-1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1],
                [-1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1],
                [-1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1],
                [-1, 1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1],
                [1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1],
                [-1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1],
                [1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1],
                [1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1],
                [-1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1],
            ]);

            let result = hankel(&input_1, &input_2);
            assert_eq!(result, expected);
        }
    }
}
