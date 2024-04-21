use ndarray::{concatenate, s, Array, Array2, Axis};
use std::{cmp::max, vec};
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
Creates a Box-Behnken design.

# Parameters

- `n`: `usize`
  The number of factors in the design.

- `center`: `usize`
  The number of center points to include. Defaults to 1 if not specified.

# Returns

- `mat`: `Array2<i32>`
  Returns the design matrix. This matrix includes all combinations of levels for each factor specified by `n`, alongside the specified number of center points.

# Example

Generate a Box-Behnken design for 3 factors with 3 center points:

```rust
// bbdesign(3, Some(3));
// Expected output:
// array([[-1.0, -1.0,  0.0],
//        [ 1.0, -1.0,  0.0],
//        [-1.0,  1.0,  0.0],
//        [ 1.0,  1.0,  0.0],
//        [-1.0,  0.0, -1.0],
//        [ 1.0,  0.0, -1.0],
//        [-1.0,  0.0,  1.0],
//        [ 1.0,  0.0,  1.0],
//        [ 0.0, -1.0, -1.0],
//        [ 0.0,  1.0, -1.0],
//        [ 0.0, -1.0,  1.0],
//        [ 0.0,  1.0,  1.0],
//        [ 0.0,  0.0,  0.0],
//        [ 0.0,  0.0,  0.0],
//        [ 0.0,  0.0,  0.0]])
```
*/
#[allow(dead_code)]
pub fn bbdesign_center(n: usize, center: usize) -> Array2<i32> {
    assert!(n >= 3, "Number of variables must be at least 3");

    let mut h_array = bb_algorithm(n);

    let center_matrix = Array2::<i32>::zeros((center, n));

    h_array = concatenate![Axis(0), h_array, center_matrix];

    h_array
}

/**
Generates a Box-Behnken design.

# Parameters

- `n`: `usize`
  The number of factors in the design. This value determines the complexity of the design matrix generated.

# Returns

- `mat`: `Array2<f64>`
  Returns the design matrix as a 2D array. The matrix represents the Box-Behnken design for the specified number of factors, with levels coded as -1, 0, and 1. The design includes all combinations of two levels at a time, with the third factor set to 0, and additional center points.

# Example

Generate a Box-Behnken design for 3 factors:

```rust
// bbdesign(3);
// Expected output:
// array([[-1.0, -1.0,  0.0],
//        [ 1.0, -1.0,  0.0],
//        [-1.0,  1.0,  0.0],
//        [ 1.0,  1.0,  0.0],
//        [-1.0,  0.0, -1.0],
//        [ 1.0,  0.0, -1.0],
//        [-1.0,  0.0,  1.0],
//        [ 1.0,  0.0,  1.0],
//        [ 0.0, -1.0, -1.0],
//        [ 0.0,  1.0, -1.0],
//        [ 0.0, -1.0,  1.0],
//        [ 0.0,  1.0,  1.0],
//        [ 0.0,  0.0,  0.0],
//        [ 0.0,  0.0,  0.0],
//        [ 0.0,  0.0,  0.0]])
```
*/
#[allow(dead_code)]
pub fn bbdesign(n: usize) -> Array2<i32> {
    assert!(n >= 3, "Number of variables must be at least 3");

    let mut h_array = bb_algorithm(n);
    let points = [0, 0, 0, 3, 3, 6, 6, 6, 8, 9, 10, 12, 12, 13, 14, 15, 16];

    let center: usize = if n <= 16 { points[n] } else { n };

    let center_matrix = Array2::<i32>::zeros((center, n));
    h_array = concatenate![Axis(0), h_array, center_matrix];

    h_array
}

/// computation algorithm for bbdesign
fn bb_algorithm(n: usize) -> Array2<i32> {
    let h_fact = super::factorial_design::ff2n(2).unwrap(); // we know this to be valid

    let nb_lines = (n * (n - 1) / 2) * h_fact.shape()[0];
    let mut h_array = Array2::<i32>::zeros((nb_lines, n));
    let mut index = 0;
    for i in 0..n - 1 {
        for j in i + 1..n {
            index += 1;
            let start_index = max(0, (index - 1) * h_fact.shape()[0]);
            let end_index = index * h_fact.shape()[0];
            for (spot, num) in [i, j].iter().enumerate() {
                let mut slice = h_array.slice_mut(s![start_index..end_index, *num]);
                let i_replacement = h_fact.slice(s![.., spot]);
                slice.assign(&i_replacement);
            }
        }
    }
    h_array
}

/**
Generates a Central Composite Design (CCD).

# Parameters

- `n`: `usize`
  The number of factors in the design.

# Optional Parameters

- `center`: `Vec<u32>`
  A vector of two integers representing the number of center points in each block of the design. Defaults to `[4, 4]`.

- `alpha`: `O&str`
  Specifies the effect of alpha on the variance. Possible values are:
  - `"orthogonal"`.
  - `"rotatable"`.

- `face`: `&str`
  Describes the relation between the start points and the corner (factorial) points. Options are:
  - `"circumscribed"``: Original form with star points at a distance `alpha` from the center.
  - `"inscribed"`: Factor settings are the star points, creating a design within those limits.
  - `"faced"`: Star points are at the center of each face of the factorial space, requiring 3 levels of each factor.

# Returns

- `mat`: `Result<Array2<f32>, String>`
  The design matrix with coded levels -1 and 1, representing the various combinations of the factors according to the CCD specifications.

# Errors

- Returns a error string if invalid `alpha` or `face` strings are provided

# Example

Generate a CCD for 3 factors:
```rust
//ccdesign(3, None, None, None);
// Expected output:
//    Array2([[-1.        , -1.        , -1.        ],
//            [ 1.        , -1.        , -1.        ],
//            [-1.        ,  1.        , -1.        ],
//            [ 1.        ,  1.        , -1.        ],
//            [-1.        , -1.        ,  1.        ],
//            [ 1.        , -1.        ,  1.        ],
//            [-1.        ,  1.        ,  1.        ],
//            [ 1.        ,  1.        ,  1.        ],
//            [ 0.        ,  0.        ,  0.        ],
//            [ 0.        ,  0.        ,  0.        ],
//            [ 0.        ,  0.        ,  0.        ],
//            [ 0.        ,  0.        ,  0.        ],
//            [-1.82574186,  0.        ,  0.        ],
//            [ 1.82574186,  0.        ,  0.        ],
//            [ 0.        , -1.82574186,  0.        ],
//            [ 0.        ,  1.82574186,  0.        ],
//            [ 0.        ,  0.        , -1.82574186],
//            [ 0.        ,  0.        ,  1.82574186],
//            [ 0.        ,  0.        ,  0.        ],
//            [ 0.        ,  0.        ,  0.        ],
//            [ 0.        ,  0.        ,  0.        ],
//            [ 0.        ,  0.        ,  0.        ]])
```
*/
#[allow(dead_code)]
pub fn ccdesign(n: usize, center: &[u32], alpha: &str, face: &str) -> Result<Array2<f32>, String> {
    if n < 2 {
        return Err("n must be 2 or higher".to_string());
    }

    let alpha = alpha.to_lowercase();
    let face = face.to_lowercase();
    let h1: Array2<f32>;
    let h2: Array2<f32>;
    let a: f32;

    match (alpha.as_str(), face.as_str()) {
        ("orthogonal", "inscribed") => {
            // Orthogonal Design
            // Inscribed CCD
            (_, a) = star(n, "orthogonal", center).unwrap();
            h1 = super::factorial_design::ff2n(n)?.mapv(|x| x as f32) / a; // Scale down the factorial points with a
            (h2, _) = star(n, "faced", &[1, 1]).unwrap();
        }
        ("rotatable", "inscribed") => {
            // Rotatable Design
            // Inscribed CCD
            (_, a) = star(n, "rotatable", &[1, 1]).unwrap();
            h1 = super::factorial_design::ff2n(n)?.mapv(|x| x as f32) / a; // Scale down the factorial points with a
            (h2, _) = star(n, "faced", &[1, 1]).unwrap();
        }
        ("orthogonal", "circumscribed") => {
            // Orthogonal Design
            // Inscribed CCD
            (h2, _) = star(n, "orthogonal", center).unwrap();
            h1 = super::factorial_design::ff2n(n)?.mapv(|x| x as f32);
        }
        ("rotatable", "circumscribed") => {
            // Rotatable Design
            // Circumscribed CCD
            (h2, _) = star(n, "rotatable", &[1, 1]).unwrap();
            h1 = super::factorial_design::ff2n(n)?.mapv(|x| x as f32);
        }
        (_, "faced") => {
            // Faced CCD
            (h2, _) = star(n, "faced", &[1, 1]).unwrap();
            h1 = super::factorial_design::ff2n(n)?.mapv(|x| x as f32);
        }
        (_, _) => {
            return Err(
                "Invalid input, alpha must be one of ['orthogonal', 'rotatable']\n
                and face must be one of ['inscribed', 'circumscribed', 'faced']"
                    .to_string(),
            )
        }
    };

    let c1 = Array2::<i32>::zeros((center[0] as usize, n)).mapv(|x| x as f32);
    let c2 = Array2::<i32>::zeros((center[1] as usize, n)).mapv(|x| x as f32);
    let h_array = concatenate![Axis(0), h1, c1, h2, c2];

    Ok(h_array)
}

/**
Generates the star points for various design matrices.

# Parameters

- `n`: `usize`
  The number of variables in the design.

# Optional Parameters

- `alpha`: `&str`
  Specifies the scaling of the star points. Available options include:
  - `"faced"`: Default. Star points are placed at the center of each face of the factorial space.
  - `"orthogonal"`: Star points are placed to preserve orthogonality.
  - `"rotatable"`: Star points are placed to achieve rotatability of the design.

- `center`: `Vec<u32>`
  A vector containing two integers that indicate the number of center points assigned in each block of the response surface design. Defaults to `[1, 1]`.

# Returns

- `Result<(Array2<f32>, f32), String>`

- `Array2<f32>` : `H`
  The star-point portion of the design matrix, positioned at `+/- alpha`.

- `f32` : `a`
  The alpha value used to scale the star points.


# Error
- Returns a error string if alpha string isn't correct

# Example

Generate star points for a 3-variable design:

```rust
// star(3, None, None);
// Expected output for H (design matrix):
// array([[-1.0,  0.0,  0.0],
//        [ 1.0,  0.0,  0.0],
//        [ 0.0, -1.0,  0.0],
//        [ 0.0,  1.0,  0.0],
//        [ 0.0,  0.0, -1.0],
//        [ 0.0,  0.0,  1.0]])
```
*/
pub fn star(n: usize, alpha: &str, center: &[u32]) -> Result<(Array2<f32>, f32), String> {
    // Star points at the center of each face of the factorial

    let a: f32;
    let alpha = alpha.to_lowercase();

    let a: f32 = match alpha.as_str() {
        "faced" => 1.0,
        "orthogonal" => {
            let nc = u32::pow(2, n as u32) as f32; // factorial points
            let nco = center[0] as f32; // center points to factorial
            let na = 2. * n as f32; // axial points
            let nao = center[1] as f32; // center points to axial design
                                        // value of alpha in orthogonal design
            let n = n as f32;
            a = (n * (1. + nao / na) / (1. + nco / nc)).sqrt();
            a
        }
        "rotatable" => {
            let nc = i32::pow(2, n as u32) as f32; // number of factorial points
            a = nc.powf(0.25); // value of alpha in rotatable design
            a
        }
        _ => return Err("Error".to_string()),
    };

    // Create the actual matrix now.
    let mut h_array: Array2<f32> = Array2::<f32>::zeros((2 * n, n));
    let arr = Array::from_vec(vec![-1.0, 1.0]);
    for i in 0..n {
        let mut slice = h_array.slice_mut(s![2 * i..2 * i + 2, i]);
        // Use assign here to copy data from `arr` into `h_array`
        slice.assign(&arr);
    }
    h_array *= a;

    Ok((h_array, a))
}

// ##############################################################################################################
// -------------------------------------------------- Tests -----------------------------------------------------
// ##############################################################################################################

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    // Import the outer module to use the function to be tested.
    use super::*;
    use ndarray::{array, Zip};

    fn array2_are_close(a: &Array2<f32>, b: &Array2<f32>, tolerance: f32) -> bool {
        // checks if all the Array2 elements are within tolerance
        Zip::from(a)
            .and(b)
            .fold(true, |acc, &a, &b| acc && (a - b).abs() <= tolerance)
    }

    #[test]
    fn bbdesign_1() {
        let input = 3;
        let expected = array![
            [-1, -1, 0],
            [1, -1, 0],
            [-1, 1, 0],
            [1, 1, 0],
            [-1, 0, -1],
            [1, 0, -1],
            [-1, 0, 1],
            [1, 0, 1],
            [0, -1, -1],
            [0, 1, -1],
            [0, -1, 1],
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ];
        assert_eq!(bbdesign(input), expected);
    }

    #[test]
    fn bbdesign_2() {
        let input = 5;
        let expected = array![
            [-1, -1, 0, 0, 0],
            [1, -1, 0, 0, 0],
            [-1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [-1, 0, -1, 0, 0],
            [1, 0, -1, 0, 0],
            [-1, 0, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [-1, 0, 0, -1, 0],
            [1, 0, 0, -1, 0],
            [-1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [-1, 0, 0, 0, -1],
            [1, 0, 0, 0, -1],
            [-1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, -1, -1, 0, 0],
            [0, 1, -1, 0, 0],
            [0, -1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, -1, 0, -1, 0],
            [0, 1, 0, -1, 0],
            [0, -1, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, -1, 0, 0, -1],
            [0, 1, 0, 0, -1],
            [0, -1, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, -1, -1, 0],
            [0, 0, 1, -1, 0],
            [0, 0, -1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, -1, 0, -1],
            [0, 0, 1, 0, -1],
            [0, 0, -1, 0, 1],
            [0, 0, 1, 0, 1],
            [0, 0, 0, -1, -1],
            [0, 0, 0, 1, -1],
            [0, 0, 0, -1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ];
        assert_eq!(bbdesign(input), expected);
    }

    #[test]
    fn bbdesign_center_1() {
        let n = 3;
        let center = 1;
        let expected = array![
            [-1, -1, 0],
            [1, -1, 0],
            [-1, 1, 0],
            [1, 1, 0],
            [-1, 0, -1],
            [1, 0, -1],
            [-1, 0, 1],
            [1, 0, 1],
            [0, -1, -1],
            [0, 1, -1],
            [0, -1, 1],
            [0, 1, 1],
            [0, 0, 0],
        ];
        assert_eq!(bbdesign_center(n, center), expected);
    }

    #[test]
    fn bbdesign_center_2() {
        let n = 4;
        let center = 5;
        let expected = array![
            [-1, -1, 0, 0],
            [1, -1, 0, 0],
            [-1, 1, 0, 0],
            [1, 1, 0, 0],
            [-1, 0, -1, 0],
            [1, 0, -1, 0],
            [-1, 0, 1, 0],
            [1, 0, 1, 0],
            [-1, 0, 0, -1],
            [1, 0, 0, -1],
            [-1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, -1, -1, 0],
            [0, 1, -1, 0],
            [0, -1, 1, 0],
            [0, 1, 1, 0],
            [0, -1, 0, -1],
            [0, 1, 0, -1],
            [0, -1, 0, 1],
            [0, 1, 0, 1],
            [0, 0, -1, -1],
            [0, 0, 1, -1],
            [0, 0, -1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ];
        assert_eq!(bbdesign_center(n, center), expected);
    }

    #[test]
    fn ccdesign_o_c() {
        let n = 2;
        let center = [2, 2];
        let alpha = "orthogonal";
        let face = "circumscribed";
        let expected: Array2<f32> = array![
            [-1., -1.],
            [1., -1.],
            [-1., 1.],
            [1., 1.],
            [0., 0.],
            [0., 0.],
            [-1.41, 0.],
            [1.414, 0.],
            [0., -1.41],
            [0., 1.414],
            [0., 0.],
            [0., 0.],
        ];
        let return_array = ccdesign(n, &center, alpha, face).unwrap();

        assert!(array2_are_close(&return_array, &expected, 0.01));
    }

    #[test]
    fn ccdesign_r_c() {
        let n = 3;
        let center = [4, 4];
        let alpha = "rotatable";
        let face = "circumscribed";
        let expected: Array2<f32> = array![
            [-1., -1., -1.],
            [1., -1., -1.],
            [-1., 1., -1.],
            [1., 1., -1.],
            [-1., -1., 1.],
            [1., -1., 1.],
            [-1., 1., 1.],
            [1., 1., 1.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [-1.68179283, 0., 0.],
            [1.68179283, 0., 0.],
            [0., -1.68179283, 0.],
            [0., 1.68179283, 0.],
            [0., 0., -1.68179283],
            [0., 0., 1.68179283],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
        ];
        let return_array = ccdesign(n, &center, alpha, face).unwrap();

        assert!(array2_are_close(&return_array, &expected, 0.01));
    }

    #[test]
    fn ccdesign_o_i() {
        let n = 4;
        let center = [5, 5];
        let alpha = "orthogonal";
        let face = "inscribed";
        let expected: Array2<f32> = array![
            [-0.44935852, -0.44935852, -0.44935852, -0.44935852],
            [0.44935852, -0.44935852, -0.44935852, -0.44935852],
            [-0.44935852, 0.44935852, -0.44935852, -0.44935852],
            [0.44935852, 0.44935852, -0.44935852, -0.44935852],
            [-0.44935852, -0.44935852, 0.44935852, -0.44935852],
            [0.44935852, -0.44935852, 0.44935852, -0.44935852],
            [-0.44935852, 0.44935852, 0.44935852, -0.44935852],
            [0.44935852, 0.44935852, 0.44935852, -0.44935852],
            [-0.44935852, -0.44935852, -0.44935852, 0.44935852],
            [0.44935852, -0.44935852, -0.44935852, 0.44935852],
            [-0.44935852, 0.44935852, -0.44935852, 0.44935852],
            [0.44935852, 0.44935852, -0.44935852, 0.44935852],
            [-0.44935852, -0.44935852, 0.44935852, 0.44935852],
            [0.44935852, -0.44935852, 0.44935852, 0.44935852],
            [-0.44935852, 0.44935852, 0.44935852, 0.44935852],
            [0.44935852, 0.44935852, 0.44935852, 0.44935852],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [-1., 0., 0., 0.],
            [1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., -1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., -1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ];
        let return_array = ccdesign(n, &center, alpha, face).unwrap();

        assert!(array2_are_close(&return_array, &expected, 0.01));
    }

    #[test]
    fn ccdesign_r_i() {
        let n = 2;
        let center = [3, 4];
        let alpha = "rotatable";
        let face = "inscribed";
        let expected: Array2<f32> = array![
            [-0.707, -0.707],
            [0.707, -0.707],
            [-0.707, 0.707],
            [0.707, 0.707],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [-1., 0.],
            [1., 0.],
            [0., -1.],
            [0., 1.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
        ];
        let return_array = ccdesign(n, &center, alpha, face).unwrap();

        assert!(array2_are_close(&return_array, &expected, 0.01));
    }

    #[test]
    fn ccdesign_f() {
        let n = 3;
        let center = [4, 1];
        let alpha = "orthogonal";
        let face = "faced";
        let expected: Array2<f32> = array![
            [-1., -1., -1.],
            [1., -1., -1.],
            [-1., 1., -1.],
            [1., 1., -1.],
            [-1., -1., 1.],
            [1., -1., 1.],
            [-1., 1., 1.],
            [1., 1., 1.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.],
            [-1., 0., 0.],
            [1., 0., 0.],
            [0., -1., 0.],
            [0., 1., 0.],
            [0., 0., -1.],
            [0., 0., 1.],
            [0., 0., 0.],
        ];
        let return_array = ccdesign(n, &center, alpha, face).unwrap();

        assert!(array2_are_close(&return_array, &expected, 0.01));
    }
}
