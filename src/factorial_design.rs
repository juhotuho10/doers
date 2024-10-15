#![allow(dead_code)]
use itertools::Itertools;
use ndarray::{concatenate, s, Array, Array1, Array2, Array3, ArrayBase, ArrayViewMut1, Axis};
use regex::Regex;
use std::collections::HashMap;

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
    Copyright (c) 2024, Juho N
    git repo: https://github.com/juhotuho10/doers
*/

/**
Creates a general full-factorial design.

# Parameters

- `levels`: `Vec<u16>`
  A vector indicating the number of levels for each input design factor.
  Each element in the vector represents a different factor and specifies how many levels that factor has.

# Returns

- `Result<Array2<u16>, String>`
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
let example_array = fullfact(&[2, 4, 3]);
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
pub fn fullfact(levels: &[u16]) -> Result<Array2<u16>, String> {
    let n = levels.len();
    let num_lines: u64 = levels.iter().map(|&level| level as u64).product();

    if num_lines > usize::MAX as u64 {
        return Err("Number of lines exceeds maximum allowable size.".to_string());
    } else if levels.iter().any(|&x| x == 0) {
        return Err("All level sizes must be 1 or higher".to_string());
    }

    let mut array: Array2<u16> = Array2::<u16>::zeros((num_lines as usize, n));

    let mut level_repeat = 1;
    let mut range_repeat = num_lines;

    for (i, &level) in levels.iter().enumerate() {
        range_repeat /= level as u64;

        // taking the slice of the array and mutating it in place without copying or moving
        let slice = array.slice_mut(s![.., i]);
        build_level(slice, level, level_repeat, range_repeat);
        level_repeat *= level;
    }
    Ok(array)
}

fn build_level(mut output: ArrayViewMut1<u16>, level: u16, repeat: u16, range_repeat: u64) {
    let mut count: usize = 0;
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

- `n`: u16
  The number of factors in the design. This determines the size and complexity of the resulting matrix.

# Returns

- `Result<Array2<i16>, String>`
  The design matrix with coded levels -1 and 1, representing the two levels for each factor across all possible combinations.

# Errors

- Returns an error string if `n` is too large (n = 64) and 2^n causes u64 to overflow

# Panics

 - will never panic despite having an unwrap

# Example

Generate a full-factorial design for 3 factors:

```rust
use doers::factorial_design::ff2n;
let example_array = ff2n(3);
//
// resulting Array2<i16>:
//
// array([[-1, -1, -1],
//        [ 1, -1, -1],
//        [-1,  1, -1],
//        [ 1,  1, -1],
//        [-1, -1,  1],
//        [ 1, -1,  1],
//        [-1,  1,  1],
//        [ 1,  1,  1]])
```
*/
pub fn ff2n(n: u16) -> Result<Array2<i16>, String> {
    let vec: Vec<u16> = vec![2; n.into()];

    let return_vec = fullfact(&vec)?;

    let return_vec = return_vec.mapv(|x| i16::try_from(x).unwrap()); // unwrap never fails since always under i16::MAX
    Ok(2 * return_vec - 1)
}

/**
Generates a Plackett-Burman design.

# Parameter

- `n`: `u32`
  The number of factors for which to create a matrix.

# Returns

- `Array2<i16>`
  Returns an orthogonal design matrix with `n` columns, one for each factor. The number of rows is the next multiple of 4 higher than `n`. For example, for 1-3 factors, there are 4 rows; for 4-7 factors, there are 8 rows, etc.

# Examples

Generate a design for 5 factors:
```rust
use doers::factorial_design::pbdesign;
let example_array = pbdesign(5);
//
// resulting Array2<i16>:
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
pub fn pbdesign(n: u32) -> Array2<i16> {
    let keep = n as usize;
    let n = n as f64;

    let n = 4. * ((n / 4.).floor() + 1.); // calculate the correct number of rows

    let mut k = 0;
    let num_array = Array::from_vec(vec![n, n / 12., n / 20.]);

    let (significand, exponents) = frexp(&num_array);

    for (index, (&mantissa, &exponent)) in significand.iter().zip(exponents.iter()).enumerate() {
        if (mantissa - 0.5).abs() < 0.01 && exponent > 0 {
            k = index;
            break;
        }
    }

    let exp = exponents[k] - 1;

    let mut h_array: Array2<i32> = match k {
        0 => Array2::ones((1, 1)),
        1 => {
            let top: Array2<i32> = Array2::ones((1, 12));

            let bottom_left: Array2<i32> = Array2::ones((11, 1));
            let bottom_right: Array2<i32> = toeplitz(
                &[-1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1],
                &[-1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],
            );

            let bottom: Array2<i32> = concatenate![Axis(1), bottom_left, bottom_right];

            concatenate![Axis(0), top, bottom]
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

            concatenate![Axis(0), top, bottom]
        }
        _ => unreachable!("Invalid value for k, this shouldn't happen"),
    };

    for _ in 0..exp {
        let h_top: Array2<i32> = concatenate![Axis(1), h_array.clone(), h_array.clone()];
        let h_bottom: Array2<i32> = concatenate![Axis(1), h_array.clone(), -h_array.clone()];
        h_array = concatenate![Axis(0), h_top, h_bottom];
    }

    // Reduce the size of the matrix as needed
    let keep = keep.min(h_array.shape()[0]);
    let h: Array2<i32> = h_array.slice(s![.., 1..=keep]).to_owned();

    // Flip the matrix upside down
    let h_flipped: Array2<i32> = h.slice(s![..;-1, ..]).to_owned();
    h_flipped.mapv(|x| x as i16)
}

/**
 Creates a Generalized Subset Design (GSD).

 # Parameters

 - `levels`: array-like
   Number of factor levels per factor in design.

 - `reduction`: u16
   Reduction factor (greater than 1). A larger `reduction` means fewer
   experiments in the design and more possible complementary designs.

 - `n_designs`: u16
   Number of complementary GSD-designs. The complementary
   designs are balanced analogous to fold-over in two-level fractional
   factorial designs.

 # Returns

 - `Result<Vec<Array2<u16>>, String>` with `n_designs` amount of complementary `Array2<u16>` matrices that have complementary designs,
    where the design size will be reduced down by reduction size

 # Errors

 - Returns a error string:
   If input is invalid or if design construction fails. The design can fail
   If any of the `levels` numbers are under 2, there will be an error.
   If `levels` has less than 2 numbers.
   if `reduction` is too large compared to values of `levels`.
   If `n_designs` numbers in under 1, the return would be empty, so we return error instead.

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

```rust
use doers::factorial_design::gsd;
let levels = [3,4,4];
let reductions = 2;
let n_arrays = 1;
let example_array = gsd(&levels, reductions, n_arrays);
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

```rust
use doers::factorial_design::gsd;
let levels = [3,3];
let reductions = 2;
let n_arrays = 2;
let example_array = gsd(&levels, reductions, n_arrays);
//
// resulting Vec<Array2<u16>>:
// [[[0, 0],
//  [0, 2],
//  [2, 0],
//  [2, 2],
//  [1, 1]],
//
//  [[0, 1],
//  [2, 1],
//  [1, 0],
//  [1, 2]]]
 ```

 If the design fails a error `String` is raised:

```rust
use doers::factorial_design::gsd;
let levels = [2,3];
let reductions = 5;
let n_arrays = 1;
let example_array = gsd(&levels, reductions, n_arrays);
 // Returns an Err: Err("Reduction is too large for the design size")
 ```

 # References

 - Surowiec, Izabella, et al. "Generalized Subset Designs in Analytical Chemistry." Analytical Chemistry 89.12 (2017): 6491-6497. <https://doi.org/10.1021/acs.analchem.7b00506>
 - Vikstrom, Ludvig, et al. Computer-implemented systems and methods for generating generalized fractional designs. US9746850 B2, filed May 9, 2014, and issued August 29, 2017. <http://www.google.se/patents/US9746850>
*/
pub fn gsd(levels: &[u16], reduction: u16, n_designs: u16) -> Result<Vec<Array2<u16>>, String> {
    if reduction < 2 {
        return Err("The level of reductions must 2 or higher".to_string());
    } else if n_designs < 1 {
        return Err("The number of designs must be 1 or higher".to_string());
    } else if levels.iter().any(|&x| x < 2) {
        return Err("All level nums must be 2 or higher".to_string());
    } else if levels.len() < 2 {
        return Err("Levels must have 2 or more numbers".to_string());
    }

    let partitions: Vec<Vec<Vec<u16>>> = make_partitions(levels, reduction);
    let latin_square: Array2<u16> = make_latin_square(reduction as usize);
    let orthogonal_arrays: Array3<u16> = make_orthogonal_arrays(&latin_square, levels.len() as u16);

    let mut design_vec: Vec<Array2<u16>> = vec![];

    for oa in orthogonal_arrays.axis_iter(Axis(0)) {
        // call the function with orthagonal arrays and forward the error if the function fails
        let design: Array2<u16> = map_partitions_to_design(&partitions, &oa.to_owned())?;
        design_vec.push(design);
    }
    // works even when n > array size
    Ok(design_vec
        .iter()
        .take(n_designs as usize)
        .cloned()
        .collect())
}

fn make_partitions(factor_levels: &[u16], num_partitions: u16) -> Vec<Vec<Vec<u16>>> {
    // Calculate total partitions and maximum size to initialize the array.
    //let max_size = *factor_levels.iter().max().unwrap_or(&1) as usize;
    let mut partitions_vec: Vec<Vec<Vec<u16>>> = vec![];

    for partition_idx in 0..num_partitions {
        let mut partition: Vec<Vec<u16>> = vec![];
        for &num_levels in factor_levels {
            let mut part: Vec<u16> = Vec::new();
            for level_i in 1..num_levels {
                let index = (partition_idx + 1) + (level_i - 1) * num_partitions;
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

fn make_orthogonal_arrays(latin_square: &Array2<u16>, n_cols: u16) -> Array3<u16> {
    let first_row = latin_square.slice(s![0, ..]);

    let mut a_matrices: Vec<Array2<u16>> = first_row
        .iter()
        .map(|&v| Array::from_elem((1, 1), v))
        .collect();

    while a_matrices[0].shape()[1] < n_cols as usize {
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
            let new_a_matrix = concatenate(
                Axis(0),
                &sub_a.iter().map(ArrayBase::view).collect::<Vec<_>>(),
            )
            .unwrap();
            new_a_matrices.push(new_a_matrix);
        }

        a_matrices = new_a_matrices;

        if a_matrices[0].shape()[1] == n_cols as usize {
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
    orthogonal_array: &Array2<u16>,
) -> Result<Array2<u16>, String> {
    let (min_value, max_value) = orthogonal_array
        .iter()
        .fold((u16::MAX, u16::MIN), |(min, max), &val| {
            (min.min(val), max.max(val))
        });

    if min_value != 0 || (max_value as usize + 1) != partitions.len() {
        return Err("Orthogonal array indexing does not match partition structure".to_string());
    }

    let mut mappings: Vec<Vec<u16>> = Vec::new();

    for row in orthogonal_array.axis_iter(Axis(0)) {
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

    // Convert mappings to Array2<u16>
    let ncols = mappings[0].len();
    let flat: Vec<u16> = mappings.into_iter().flatten().collect();
    let nrows = flat.len() / ncols;

    Ok(Array2::from_shape_vec((nrows, ncols), flat).unwrap() - 1)
}

/**
Create a 2-level fractional-factorial design with a generator string.

# Parameter

- `design`: `&str`:
  A string, consisting of lowercase, uppercase letters or a negative operator "-"
  though all charactes will be changed into lower case

# Returns

- `Array2<i16>`
  ff2n design of the individual characters as well as the combinatory designs for the characters

# Notes

The argument `&str` defines the main factors of the experiment and the factors
whose levels are the products of the main factors. For example, if

let design = "a b ab"

then "a" and "b" are the main factors, while the 3rd factor is the product of the first two.

# Panics

Should never panic despite having expect and unwrap

# Examples

Generate a conditional design where we have designs for a b and c
but also have a conditional one that is the negative product of a and b

```rust
use doers::factorial_design::fracfact;
let example_array = fracfact("a b c -ab");
//
// resulting `Array2<i16>`:
//
//          a   b   c  -ab
// array([[-1, -1, -1, -1],
//        [ 1, -1, -1,  1],
//        [-1,  1, -1,  1],
//        [ 1,  1, -1, -1],
//        [-1, -1,  1, -1],
//        [ 1, -1,  1,  1],
//        [-1,  1,  1,  1],
//        [ 1,  1,  1, -1]])
```
*/
pub fn fracfact(design: &str) -> Array2<i16> {
    let design = design.to_lowercase();
    let design = design.as_str();

    // separate letters and remove "-" and "+" from them
    let separator_regex = Regex::new(r"\+|\s|\-").expect("regex error"); // hardcoded regex wont panic
    let char_splits: Vec<&str> = separator_regex
        .split(design)
        .filter(|x| !x.is_empty())
        .collect();

    // indexes of letters main factor letters (singular letters)
    let single_letter_i = char_splits
        .iter()
        .enumerate()
        .filter_map(|(i, &x)| if x.len() == 1 { Some(i) } else { None })
        .collect_vec();

    // indexes of letter combinations
    let multi_letter_i = char_splits
        .iter()
        .enumerate()
        .filter_map(|(i, &x)| if x.len() == 1 { None } else { Some(i) })
        .collect_vec();

    // new splits with the "+" and "-" included with the letters
    let separator_regex = Regex::new(r"\s").expect("regex error"); // hardcoded regex wont panic
    let splits: Vec<&str> = separator_regex.split(design).collect();

    // indexes that are marked negative
    let minus_i = splits
        .iter()
        .enumerate()
        .filter_map(|(i, &x)| if x.contains('-') { Some(i) } else { None })
        .collect_vec();

    // ff2n design of the main factors
    let h_single = ff2n(single_letter_i.len() as u16).expect("too many letters"); // single letters will never be over the len needed for ff2n panics

    // assign the designs at the correct indexes in the final design
    let mut h_combined: Array2<i16> = Array::zeros((h_single.shape()[0], splits.len()));
    for (array, i) in h_single.axis_iter(Axis(1)).zip(&single_letter_i) {
        let mut into = h_combined.slice_mut(s![.., *i]);
        into.assign(&array);
    }

    // creating a map to map the main factor single chars to their indexes in the main design
    let mut char_to_i: HashMap<char, usize> = HashMap::new();
    for &i in &single_letter_i {
        if let Some(first_char) = char_splits[i].chars().next() {
            char_to_i.insert(first_char, i);
        }
    }

    for i in multi_letter_i {
        let combination = char_splits[i];
        let letters = combination.chars().collect_vec();

        // indices for all the individual chars in letter combiantions to get their design from the h_combined
        let indices = letters.iter().map(|x| char_to_i[x]).collect_vec();

        // multiply all the designs together
        let mut multiplied = Array1::ones(h_combined.shape()[0]);

        for index in indices {
            let other_slice = h_combined.slice(s![.., index]).to_owned();
            multiplied = multiplied * other_slice;
        }

        h_combined.slice_mut(s![.., i]).assign(&multiplied);
    }

    // apply the "-" to make correct columns negative
    for i in minus_i {
        let mut slice = h_combined.slice_mut(s![.., i]);
        slice.mapv_inplace(|x| -x);
    }

    h_combined
}

// ##############################################################################################################
// -------------------------------------------------- Utils -----------------------------------------------------
// ##############################################################################################################

/**
frexp for a Array1 of f64
Decomposes elements of a float array into mantissas and exponents.

Each float `x` is transformed into `m * 2^e`, where `m` is the mantissa and `e` the exponent.

# Arguments
`Array1<f64>`

# Returns
A tuple of arrays (`Array1<f64>`, `Array1<i32>`) for mantissas and exponents respectively.
*/
fn frexp(arr: &Array1<f64>) -> (Array1<f64>, Array1<i32>) {
    let mantissas = arr.mapv(|x| {
        if x == 0.0 {
            0.0 // Mantissa for zero
        } else {
            let mut y = x.abs();
            let exponent = y.log2().floor() + 1.0;
            y /= f64::powf(2.0, exponent);
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
    use ndarray::{ArrayBase, Data, Dimension, Zip};

    fn arrays_are_close<S, D>(a: &ArrayBase<S, D>, b: &ArrayBase<S, D>, tolerance: f64) -> bool
    // checks if all the Array1 elements are within tolerance
    where
        S: Data<Elem = f64>,
        D: Dimension,
    {
        assert_eq!(a.shape(), b.shape(), "array shapes must be the same");
        Zip::from(a)
            .and(b)
            .fold(true, |acc, &a, &b| acc && (a - b).abs() <= tolerance)
    }

    mod functionality_tests {
        use ndarray::array;

        use super::*;

        #[test]
        fn fullfact_1() {
            let input = vec![];
            let expected = array![[]];
            assert_eq!(fullfact(&input).unwrap(), expected);
        }

        #[test]
        fn fullfact_2() {
            let input = vec![1, 2];
            let expected = array![[0, 0], [0, 1]];
            assert_eq!(fullfact(&input).unwrap(), expected);
        }

        #[test]
        fn fullfact_3() {
            let input = vec![1, 2, 3];
            let expected = array![
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 1, 1],
                [0, 0, 2],
                [0, 1, 2],
            ];
            assert_eq!(fullfact(&input).unwrap(), expected);
        }

        #[test]
        fn ff2n_1() {
            let input = 1;
            let expected = array![[-1], [1]];
            assert_eq!(ff2n(input).unwrap(), expected);
        }

        #[test]
        fn ff2n_2() {
            let input = 2;
            let expected = array![[-1, -1], [1, -1], [-1, 1], [1, 1]];
            assert_eq!(ff2n(input).unwrap(), expected);
        }

        #[test]
        fn ff2n_3() {
            let input = 4;
            let expected = array![
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
            ];
            assert_eq!(ff2n(input).unwrap(), expected);
        }

        #[test]
        fn pbdesign_1() {
            // uses k=0
            let input = 2;
            let expected = array![[-1, -1], [1, -1], [-1, 1], [1, 1]];
            assert_eq!(pbdesign(input), expected);
        }

        #[test]
        fn pbdesign_2() {
            // uses k=0
            let input = 4;
            let expected = array![
                [-1, -1, 1, -1],
                [1, -1, -1, -1],
                [-1, 1, -1, -1],
                [1, 1, 1, -1],
                [-1, -1, 1, 1],
                [1, -1, -1, 1],
                [-1, 1, -1, 1],
                [1, 1, 1, 1],
            ];
            assert_eq!(pbdesign(input), expected);
        }

        #[test]
        fn pbdesign_3() {
            // uses k=1
            let input = 8;
            let expected = array![
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
            ];
            assert_eq!(pbdesign(input), expected);
        }

        #[test]
        fn pbdesign_4() {
            // uses k=2
            let input = 16;
            let expected = array![
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
            ];
            assert_eq!(pbdesign(input), expected);
        }

        #[test]
        fn gsd_1() {
            let levels = [2, 2, 3];
            let reductions = 2;
            let n = 1;
            let expected = vec![array![
                [0, 0, 0],
                [0, 0, 2],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 2],
            ]];
            assert_eq!(gsd(&levels, reductions, n), Ok(expected));
        }

        #[test]
        fn gsd_2() {
            let levels = [3, 3, 2];
            let reductions = 4;
            let n = 2;
            let expected = vec![
                array![[0, 0, 0], [0, 1, 1], [1, 0, 1], [2, 2, 0]],
                array![[0, 0, 1], [1, 2, 0], [2, 1, 0], [2, 2, 1]],
            ];
            assert_eq!(gsd(&levels, reductions, n), Ok(expected));
        }

        #[test]
        fn gsd_3() {
            let levels = [3, 3, 3];
            let reductions = 3;
            let n = 2;
            let expected = vec![
                array![
                    [0, 0, 0],
                    [0, 1, 1],
                    [0, 2, 2],
                    [1, 0, 1],
                    [1, 1, 2],
                    [1, 2, 0],
                    [2, 0, 2],
                    [2, 1, 0],
                    [2, 2, 1],
                ],
                array![
                    [0, 0, 1],
                    [0, 1, 2],
                    [0, 2, 0],
                    [1, 0, 2],
                    [1, 1, 0],
                    [1, 2, 1],
                    [2, 0, 0],
                    [2, 1, 1],
                    [2, 2, 2],
                ],
            ];
            assert_eq!(gsd(&levels, reductions, n), Ok(expected));
        }

        #[test]
        fn fracfact_1() {
            let input = "";
            let expected = array![[0]];
            assert_eq!(fracfact(input), expected);
        }

        #[test]
        fn fracfact_2() {
            let input = "a b c";
            let expected = array![
                [-1, -1, -1],
                [1, -1, -1],
                [-1, 1, -1],
                [1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [-1, 1, 1],
                [1, 1, 1],
            ];
            assert_eq!(fracfact(input), expected);
        }

        #[test]
        fn fracfact_3() {
            let input = "A B C -C";
            let expected = array![
                [-1, -1, -1, 1],
                [1, -1, -1, 1],
                [-1, 1, -1, 1],
                [1, 1, -1, 1],
                [-1, -1, 1, 1],
                [1, -1, 1, 1],
                [-1, 1, 1, 1],
                [1, 1, 1, 1],
                [-1, -1, -1, -1],
                [1, -1, -1, -1],
                [-1, 1, -1, -1],
                [1, 1, -1, -1],
                [-1, -1, 1, -1],
                [1, -1, 1, -1],
                [-1, 1, 1, -1],
                [1, 1, 1, -1],
            ];
            assert_eq!(fracfact(input), expected);
        }

        #[test]
        fn fracfact_4() {
            let input = "a b c ab ac";
            let expected = array![
                [-1, -1, -1, 1, 1],
                [1, -1, -1, -1, -1],
                [-1, 1, -1, -1, 1],
                [1, 1, -1, 1, -1],
                [-1, -1, 1, 1, -1],
                [1, -1, 1, -1, 1],
                [-1, 1, 1, -1, -1],
                [1, 1, 1, 1, 1],
            ];
            assert_eq!(fracfact(input), expected);
        }

        #[test]
        fn fracfact_5() {
            let input = "a b -c ab -ac";
            let expected = array![
                [-1, -1, 1, 1, -1],
                [1, -1, 1, -1, 1],
                [-1, 1, 1, -1, -1],
                [1, 1, 1, 1, 1],
                [-1, -1, -1, 1, 1],
                [1, -1, -1, -1, -1],
                [-1, 1, -1, -1, 1],
                [1, 1, -1, 1, -1],
            ];
            assert_eq!(fracfact(input), expected);
        }

        #[test]
        fn fracfact_6() {
            let input = "a -b -abc c ac d -adc";
            let expected = array![
                [-1, 1, 1, -1, 1, -1, 1],
                [1, 1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, 1, -1, 1],
                [1, -1, 1, -1, -1, -1, -1],
                [-1, 1, -1, 1, -1, -1, -1],
                [1, 1, 1, 1, 1, -1, 1],
                [-1, -1, 1, 1, -1, -1, -1],
                [1, -1, -1, 1, 1, -1, 1],
                [-1, 1, 1, -1, 1, 1, -1],
                [1, 1, -1, -1, -1, 1, 1],
                [-1, -1, -1, -1, 1, 1, -1],
                [1, -1, 1, -1, -1, 1, 1],
                [-1, 1, -1, 1, -1, 1, 1],
                [1, 1, 1, 1, 1, 1, -1],
                [-1, -1, 1, 1, -1, 1, 1],
                [1, -1, -1, 1, 1, 1, -1],
            ];
            assert_eq!(fracfact(input), expected);
        }
    }

    mod utilities_tests {
        use ndarray::{array, Array1};

        use crate::factorial_design::{frexp, hankel, tests::arrays_are_close, toeplitz};

        #[test]
        fn frexp_test() {
            let levels: Array1<f64> = array![5., 5. / 12., 5. / 20.];
            let (f, e) = frexp(&levels);

            let e = e.mapv(|x| x as f64);

            let expected_f = array![0.625, 0.83333, 0.5];
            let expected_e = array![3., -1., -1.];
            assert!(
                arrays_are_close(&f, &expected_f, 0.01),
                "array values do not match"
            );
            assert!(
                arrays_are_close(&e, &expected_e, 0.01),
                "array values do not match"
            );
        }

        #[test]
        fn toeplitz_test() {
            let input_1 = [-1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1];
            let input_2 = [-1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1];
            let expected = array![
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
            ];

            let result = toeplitz(&input_1, &input_2);

            assert_eq!(result, expected);
        }

        #[test]
        fn hankel_test() {
            let input_1 = [-1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, -1];
            let input_2 = [1, -1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, 1];
            let expected = array![
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
            ];

            let result = hankel(&input_1, &input_2);
            assert_eq!(result, expected);
        }
    }
}
