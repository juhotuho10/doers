use ndarray::{array, concatenate, s, Array2, Axis};

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

// Enum for star() function for the alpha variable
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Alpha {
    Orthogonal,
    Rotatable,
    Faced,
}
// Enum for star() function for the Face variable
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Face {
    Inscribed,
    Circumscribed,
}

/**
Creates a Box-Behnken design.

# Parameters

- `n`: `usize`
  The number of factors in the design.

- `center`: `usize`
  The number of center points to include.

# Returns

- `mat`: `Result<Array2<i32>, String>`
  Returns the design matrix. This matrix includes all combinations of levels for each factor specified by `n`, alongside the specified number of center points.

# Errors
- Will return an error string if the `n` is less than 3

# Example

Generate a Box-Behnken design for 3 factors with 3 center points:

```rust
use doers::response_surface_design::bbdesign_center;
let n = 3;
let center = 3;
let output = bbdesign_center(n, center);

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
pub fn bbdesign_center(n: usize, center: usize) -> Result<Array2<i32>, String> {
    if n < 3 {
        return Err("The number of factors in the design (n) must be 3 or higher".to_owned());
    }

    let mut h_array = bb_algorithm(n);

    let center_matrix = Array2::<i32>::zeros((center, n));

    h_array = concatenate![Axis(0), h_array, center_matrix];

    Ok(h_array)
}

/**
Generates a Box-Behnken design.

# Parameters

- `n`: `usize`
  The number of factors in the design. This value determines the complexity of the design matrix generated.

# Returns

- `mat`: `Array2<f64>`
  Returns the design matrix as a 2D array. The matrix represents the Box-Behnken design for the specified number of factors, with levels coded as -1, 0, and 1. The design includes all combinations of two levels at a time, with the third factor set to 0, and additional center points.

# Errors

- Returns a error string if n is too small


# Example

Generate a Box-Behnken design for 3 factors:

```rust
use doers::response_surface_design::bbdesign;
let n = 3;
let output = bbdesign(n);
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
pub fn bbdesign(n: usize) -> Result<Array2<i32>, String> {
    if n < 3 {
        return Err("The number of factors in the design (n) must be 3 or higher".to_owned());
    }

    let mut h_array = bb_algorithm(n);
    let points = [0, 0, 0, 3, 3, 6, 6, 6, 8, 9, 10, 12, 12, 13, 14, 15, 16];

    let center: usize = if n <= 16 { points[n] } else { n };

    let center_matrix = Array2::<i32>::zeros((center, n));
    h_array = concatenate![Axis(0), h_array, center_matrix];

    Ok(h_array)
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
            let maybe_start_index = (index - 1) * h_fact.shape()[0];
            let start_index = 0.max(maybe_start_index);
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

- `center`: `&[u32]`
  A vector of two integers representing the number of center points in each block of the design.

- `alpha`: `Alpha`
  Specifies the effect of alpha on the variance. Possible values are:
  - `Alpha::Orthogonal`.
  - `Alpha::Rotatable`.
  - `Alpha::Faced`: Star points are at the center of each face of the factorial space, requiring 3 levels of each factor.

- `face`: `Face`
  Describes the relation between the start points and the corner (factorial) points. Options are:
  - `Face::Circumscribed`: Original form with star points at a distance `alpha` from the center.
  - `Face::Inscribed`: Factor settings are the star points, creating a design within those limits.

# Returns

- `mat`: `Result<Array2<f32>, String>`
  The design matrix with coded levels -1 and 1, representing the various combinations of the factors according to the CCD specifications.

# Errors

- Returns a error string if n is less than 2

# Example

Generate a CCD for 3 factors:
```rust
use doers::response_surface_design::{Alpha, Face, ccdesign};

let n = 2;
let center = [3, 2];
let alpha = Alpha::Orthogonal;
let face = Face::Circumscribed;
let output = ccdesign(n, &center, alpha, face);
// Expected output:
//    Array2([[-1.0,       -1.0      ],
//            [ 1.0,       -1.0      ],
//            [-1.0,        1.0      ],
//            [ 1.0,        1.0      ],
//            [ 0.0,        0.0      ],
//            [ 0.0,        0.0      ],
//            [ 0.0,        0.0      ],
//            [-1.3093073,  0.0      ],
//            [ 1.3093073,  0.0      ],
//            [ 0.0,       -1.3093073],
//            [ 0.0,        1.3093073],
//            [ 0.0,        0.0      ],
//            [ 0.0,        0.0      ]],)
```
*/
#[allow(dead_code, clippy::cast_precision_loss)] // ff2n values will never lose precision with f32
pub fn ccdesign(n: usize, center: &[u32], alpha: Alpha, face: Face) -> Result<Array2<f32>, String> {
    use super::factorial_design::ff2n;

    if n < 2 {
        return Err("The number of factors in the design (n) must be 2 or higher".to_owned());
    }

    let h1: Array2<f32>;
    let h2: Array2<f32>;
    let a: f32;

    match (&alpha, face) {
        (Alpha::Orthogonal, Face::Inscribed) => {
            // Orthogonal Design
            // Inscribed CCD
            (_, a) = star(n, alpha, center);
            h1 = ff2n(n)?.mapv(|x| x as f32) / a; // Scale down the factorial points with a
            (h2, _) = star(n, Alpha::Faced, &[1, 1]);
        }
        (Alpha::Rotatable, Face::Inscribed) => {
            // Rotatable Design
            // Inscribed CCD
            (_, a) = star(n, alpha, &[1, 1]);
            h1 = ff2n(n)?.mapv(|x| x as f32) / a; // Scale down the factorial points with a
            (h2, _) = star(n, Alpha::Faced, &[1, 1]);
        }
        (Alpha::Orthogonal, Face::Circumscribed) => {
            // Orthogonal Design
            // Inscribed CCD
            h1 = ff2n(n)?.mapv(|x| x as f32);
            (h2, _) = star(n, alpha, center);
        }
        (Alpha::Rotatable, Face::Circumscribed) => {
            // Rotatable Design
            // Circumscribed CCD
            h1 = ff2n(n)?.mapv(|x| x as f32);
            (h2, _) = star(n, alpha, &[1, 1]);
        }
        (Alpha::Faced, _) => {
            // Faced Design
            h1 = ff2n(n)?.mapv(|x| x as f32);
            (h2, _) = star(n, alpha, &[1, 1]);
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

- `alpha`: `&str`
  Specifies the scaling of the star points. Available options include:
  - `Alpha::Faced`: Star points are placed at the center of each face of the factorial space.
  - `Alpha::Orthogonal`: Star points are placed to preserve orthogonality.
  - `Alpha::Rotatable`: Star points are placed to achieve rotatability of the design.

- `center`: `&[u32]`
  A vector containing two integers that indicate the number of center points assigned in each block of the response surface design.

# Returns

- `(Array2<f32>, f32)`

- `Array2<f32>` : `H`
  The star-point portion of the design matrix, positioned at `+/- alpha`.

- `f32` : `a`
  The alpha value used to scale the star points.

# Example

Generate star points for a 2-variable design:

```rust
use doers::response_surface_design::{Alpha, star};
let n = 2;
let center = [4, 2];
let alpha = Alpha::Orthogonal;
let (output, a) = star(n, alpha, &center);
// Expected output for H (design matrix):
// array([[-1.2247449, 0.0      ],
//         [ 1.2247449, 0.0      ],
//         [ 0.0,      -1.2247449],
//         [ 0.0,       1.2247449]])
//
// Expected a:
// 1.2247449


```
*/
pub fn star(n: usize, alpha: Alpha, center: &[u32]) -> (Array2<f32>, f32) {
    // Star points at the center of each face of the factorial

    let a: f64 = match alpha {
        Alpha::Faced => 1.0,
        Alpha::Orthogonal => {
            let nc = u32::pow(2, n as u32) as f64; // factorial points
            let nco = center[0] as f64; // center points to factorial
            let na = 2. * n as f64; // axial points
            let nao = center[1] as f64; // center points to axial design
                                        // value of alpha in orthogonal design
            let n = n as f64;
            (n * (1. + nao / na) / (1. + nco / nc)).sqrt()
        }
        Alpha::Rotatable => {
            let nc = i32::pow(2, n as u32) as f64; // number of factorial points
            nc.powf(0.25) // value of alpha in rotatable design
        }
    };

    // Create the actual matrix now.
    let mut h_array: Array2<f64> = Array2::<f64>::zeros((2 * n, n));
    let arr = array![-1.0, 1.0];
    for i in 0..n {
        let index = 2 * i;
        let mut slice = h_array.slice_mut(s![index..index + 2, i]);
        // Use assign here to copy data from `arr` into `h_array`
        slice.assign(&arr);
    }
    h_array *= a;
    let h_array = h_array.mapv(|x| x as f32);
    let a = a as f32;

    (h_array, a)
}

// ##############################################################################################################
// -------------------------------------------------- Tests -----------------------------------------------------
// ##############################################################################################################

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    // Import the outer module to use the function to be tested.
    use super::*;
    use ndarray::{array, ArrayBase, Data, Dimension, Zip};

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

        let return_array = bbdesign(input).unwrap();
        assert_eq!(return_array, expected, "arrays aren't equal");
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

        let return_array = bbdesign(input).unwrap();
        assert_eq!(return_array, expected, "arrays aren't equal");
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

        let return_array = bbdesign_center(n, center).unwrap(); // can't fail
        assert_eq!(return_array, expected, "arrays aren't equal");
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

        let return_array = bbdesign_center(n, center).unwrap(); // can't fail
        assert_eq!(return_array, expected, "arrays aren't equal");
    }

    #[test]
    fn ccdesign_o_c() {
        let n = 2;
        let center = [4, 2];
        let alpha = Alpha::Orthogonal;
        let face = Face::Circumscribed;
        let expected: Array2<f32> = array![
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [-1.22474487, 0.0],
            [1.22474487, 0.0],
            [0.0, -1.22474487],
            [0.0, 1.22474487],
            [0.0, 0.0],
            [0.0, 0.0],
        ];
        let return_array = ccdesign(n, &center, alpha, face).unwrap();

        assert_eq!(
            return_array.shape(),
            expected.shape(),
            "array shapes do not match, {:?} vs expected: {:?}",
            return_array.shape(),
            expected.shape()
        );

        assert!(
            arrays_are_close(&return_array, &expected, 0.01),
            "array values do not match"
        );
    }

    #[test]
    fn ccdesign_o_c_2() {
        let n = 2;
        let center = [7, 1];
        let alpha = Alpha::Orthogonal;
        let face = Face::Circumscribed;
        let expected: Array2<f32> = array![
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [-0.953462, 0.0],
            [0.953462, 0.0],
            [0.0, -0.953462],
            [0.0, 0.953462],
            [0.0, 0.0],
        ];
        let return_array = ccdesign(n, &center, alpha, face).unwrap();

        assert_eq!(
            return_array.shape(),
            expected.shape(),
            "array shapes do not match, {:?} vs expected: {:?}",
            return_array.shape(),
            expected.shape()
        );
        assert!(
            arrays_are_close(&return_array, &expected, 0.01),
            "array values do not match"
        );
    }

    #[test]
    fn ccdesign_r_c() {
        let n = 3;
        let center = [8, 3];
        let alpha = Alpha::Rotatable;
        let face = Face::Circumscribed;
        let expected: Array2<f32> = array![
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.681792, 0.0, 0.0],
            [1.681792, 0.0, 0.0],
            [0.0, -1.681792, 0.0],
            [0.0, 1.681792, 0.0],
            [0.0, 0.0, -1.681792],
            [0.0, 0.0, 1.681792],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let return_array = ccdesign(n, &center, alpha, face).unwrap();

        assert_eq!(
            return_array.shape(),
            expected.shape(),
            "array shapes do not match, {:?} vs expected: {:?}",
            return_array.shape(),
            expected.shape()
        );
        assert!(
            arrays_are_close(&return_array, &expected, 0.01),
            "array values do not match"
        );
    }

    #[test]
    fn ccdesign_r_c_2() {
        let n = 5;
        let center = [15, 19];
        let alpha = Alpha::Rotatable;
        let face = Face::Circumscribed;
        let expected: Array2<f32> = array![
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-2.378414, 0.0, 0.0, 0.0, 0.0],
            [2.378414, 0.0, 0.0, 0.0, 0.0],
            [0.0, -2.378414, 0.0, 0.0, 0.0],
            [0.0, 2.378414, 0.0, 0.0, 0.0],
            [0.0, 0.0, -2.378414, 0.0, 0.0],
            [0.0, 0.0, 2.378414, 0.0, 0.0],
            [0.0, 0.0, 0.0, -2.378414, 0.0],
            [0.0, 0.0, 0.0, 2.378414, 0.0],
            [0.0, 0.0, 0.0, 0.0, -2.378414],
            [0.0, 0.0, 0.0, 0.0, 2.378414],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let return_array = ccdesign(n, &center, alpha, face).unwrap();

        assert_eq!(
            return_array.shape(),
            expected.shape(),
            "array shapes do not match, {:?} vs expected: {:?}",
            return_array.shape(),
            expected.shape()
        );
        assert!(
            arrays_are_close(&return_array, &expected, 0.01),
            "array values do not match"
        );
    }

    #[test]
    fn ccdesign_o_i() {
        let n = 4;
        let center = [7, 1];
        let alpha = Alpha::Orthogonal;
        let face = Face::Inscribed;
        let expected: Array2<f32> = array![
            [-0.56519, -0.56519, -0.56519, -0.56519],
            [0.56519, -0.56519, -0.56519, -0.56519],
            [-0.56519, 0.56519, -0.56519, -0.56519],
            [0.56519, 0.56519, -0.56519, -0.56519],
            [-0.56519, -0.56519, 0.56519, -0.56519],
            [0.56519, -0.56519, 0.56519, -0.56519],
            [-0.56519, 0.56519, 0.56519, -0.56519],
            [0.56519, 0.56519, 0.56519, -0.56519],
            [-0.56519, -0.56519, -0.56519, 0.56519],
            [0.56519, -0.56519, -0.56519, 0.56519],
            [-0.56519, 0.56519, -0.56519, 0.56519],
            [0.56519, 0.56519, -0.56519, 0.56519],
            [-0.56519, -0.56519, 0.56519, 0.56519],
            [0.56519, -0.56519, 0.56519, 0.56519],
            [-0.56519, 0.56519, 0.56519, 0.56519],
            [0.56519, 0.56519, 0.56519, 0.56519],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let return_array = ccdesign(n, &center, alpha, face).unwrap();

        assert_eq!(
            return_array.shape(),
            expected.shape(),
            "array shapes do not match, {:?} vs expected: {:?}",
            return_array.shape(),
            expected.shape()
        );
        assert!(
            arrays_are_close(&return_array, &expected, 0.01),
            "array values do not match"
        );
    }

    #[test]
    fn ccdesign_r_i() {
        let n = 2;
        let center = [3, 4];
        let alpha = Alpha::Rotatable;
        let face = Face::Inscribed;
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

        assert_eq!(
            return_array.shape(),
            expected.shape(),
            "array shapes do not match, {:?} vs expected: {:?}",
            return_array.shape(),
            expected.shape()
        );
        assert!(
            arrays_are_close(&return_array, &expected, 0.01),
            "array values do not match"
        );
    }

    #[test]
    fn ccdesign_f() {
        let n = 3;
        let center = [10, 6];
        let alpha = Alpha::Faced;
        let face = Face::Inscribed;
        let expected: Array2<f32> = array![
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let return_array = ccdesign(n, &center, alpha, face).unwrap();

        assert_eq!(
            return_array.shape(),
            expected.shape(),
            "array shapes do not match, {:?} vs expected: {:?}",
            return_array.shape(),
            expected.shape()
        );
        assert!(
            arrays_are_close(&return_array, &expected, 0.01),
            "array values do not match"
        );

        let n = 3;
        let center = [10, 6];
        let alpha = Alpha::Faced;
        let face = Face::Circumscribed;

        let return_array = ccdesign(n, &center, alpha, face).unwrap();

        assert_eq!(
            return_array.shape(),
            expected.shape(),
            "array shapes do not match, {:?} vs expected: {:?}",
            return_array.shape(),
            expected.shape()
        );
        assert!(
            arrays_are_close(&return_array, &expected, 0.01),
            "array values do not match"
        );
    }
}
