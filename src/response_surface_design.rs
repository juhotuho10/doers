use ndarray::{concatenate, s, Array, Array2, Axis};
use std::{cmp::max, vec};

/**
Create a Box-Behnken design

Parameters
----------
n : usize
    The number of factors in the design
--------
center : usize
    The number of center points to include (default = 1).

Returns
-------
mat : 2d-array
    The design matrix

Example
-------
::

    >>> bbdesign(3, 3)
    array([[-1., -1.,  0.],
            [ 1., -1.,  0.],
            [-1.,  1.,  0.],
            [ 1.,  1.,  0.],
            [-1.,  0., -1.],
            [ 1.,  0., -1.],
            [-1.,  0.,  1.],
            [ 1.,  0.,  1.],
            [ 0., -1., -1.],
            [ 0.,  1., -1.],
            [ 0., -1.,  1.],
            [ 0.,  1.,  1.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])

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
Create a Box-Behnken design

Parameters
----------
n : int
    The number of factors in the design

Returns
-------
mat : 2d-array
    The design matrix

Example
-------
::

    >>> bbdesign(3)
    array([[-1., -1.,  0.],
            [ 1., -1.,  0.],
            [-1.,  1.,  0.],
            [ 1.,  1.,  0.],
            [-1.,  0., -1.],
            [ 1.,  0., -1.],
            [-1.,  0.,  1.],
            [ 1.,  0.,  1.],
            [ 0., -1., -1.],
            [ 0.,  1., -1.],
            [ 0., -1.,  1.],
            [ 0.,  1.,  1.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.],
            [ 0.,  0.,  0.]])

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

fn bb_algorithm(n: usize) -> Array2<i32> {
    //First, compute a factorial DOE with 2 parameters
    let h_fact = super::factorial_design::ff2n(2).unwrap(); // we know this to be valid

    // Now we populate the real DOE with this DOE

    // We made a factorial design on each pair of dimensions
    // - So, we created a factorial design with two factors
    // - Make two loops
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
Central composite design

Parameters
----------
n : int
    The number of factors in the design.

Optional
--------
center : int array
    A 1-by-2 array of integers, the number of center points in each block
    of the design. (Default: (4, 4)).
alpha : str
    A string describing the effect of alpha has on the variance. ``alpha``
    can take on the following values:

    1. 'orthogonal' or 'o' (Default)

    2. 'rotatable' or 'r'

face : str
    The relation between the start points and the corner (factorial) points.
    There are three options for this input:

    1. 'circumscribed' or 'ccc': This is the original form of the central
        composite design. The star points are at some distance ``alpha``
        from the center, based on the properties desired for the design.
        The start points establish new extremes for the low and high
        settings for all factors. These designs have circular, spherical,
        or hyperspherical symmetry and require 5 levels for each factor.
        Augmenting an existing factorial or resolution V fractional
        factorial design with star points can produce this design.

    2. 'inscribed' or 'cci': For those situations in which the limits
        specified for factor settings are truly limits, the CCI design
        uses the factors settings as the star points and creates a factorial
        or fractional factorial design within those limits (in other words,
        a CCI design is a scaled down CCC design with each factor level of
        the CCC design divided by ``alpha`` to generate the CCI design).
        This design also requires 5 levels of each factor.

    3. 'faced' or 'ccf': In this design, the star points are at the center
        of each face of the factorial space, so ``alpha`` = 1. This
        variety requires 3 levels of each factor. Augmenting an existing
        factorial or resolution V design with appropriate star points can
        also produce this design.

Notes
-----
- Fractional factorial designs are not (yet) available here.
- 'ccc' and 'cci' can be rotatable design, but 'ccf' cannot.

Returns
-------
mat : 2d-array
    The design matrix with coded levels -1 and 1

Example
-------
::

    >>> ccdesign(3)
    array([[-1.        , -1.        , -1.        ],
            [ 1.        , -1.        , -1.        ],
            [-1.        ,  1.        , -1.        ],
            [ 1.        ,  1.        , -1.        ],
            [-1.        , -1.        ,  1.        ],
            [ 1.        , -1.        ,  1.        ],
            [-1.        ,  1.        ,  1.        ],
            [ 1.        ,  1.        ,  1.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [-1.82574186,  0.        ,  0.        ],
            [ 1.82574186,  0.        ,  0.        ],
            [ 0.        , -1.82574186,  0.        ],
            [ 0.        ,  1.82574186,  0.        ],
            [ 0.        ,  0.        , -1.82574186],
            [ 0.        ,  0.        ,  1.82574186],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ]])

*/
#[allow(dead_code)]
pub fn ccdesign(
    n: usize,
    center: Vec<u32>,
    alpha: &str,
    face: &str,
) -> Result<Array2<f32>, String> {
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
            (_, a) = star(n, "orthogonal", center.clone()).unwrap();
            h1 = super::factorial_design::ff2n(n)?.mapv(|x| x as f32) / a; // Scale down the factorial points with a
            (h2, _) = star(n, "faced", vec![1, 1]).unwrap();
        }
        ("rotatable", "inscribed") => {
            // Rotatable Design
            // Inscribed CCD
            (_, a) = star(n, "rotatable", vec![1, 1]).unwrap();
            h1 = super::factorial_design::ff2n(n)?.mapv(|x| x as f32) / a; // Scale down the factorial points with a
            (h2, _) = star(n, "faced", vec![1, 1]).unwrap();
        }
        ("orthogonal", "circumscribed") => {
            // Orthogonal Design
            // Inscribed CCD
            (h2, _) = star(n, "orthogonal", center.clone()).unwrap();
            h1 = super::factorial_design::ff2n(n)?.mapv(|x| x as f32);
        }
        ("rotatable", "circumscribed") => {
            // Rotatable Design
            // Circumscribed CCD
            (h2, _) = star(n, "rotatable", vec![1, 1]).unwrap();
            h1 = super::factorial_design::ff2n(n)?.mapv(|x| x as f32);
        }
        (_, "faced") => {
            // Faced CCD
            (h2, _) = star(n, "faced", vec![1, 1]).unwrap();
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
Create the star points of various design matrices

Parameters
----------
n : int
    The number of variables in the design

Optional
--------
alpha : str
    Available values are 'faced' (default), 'orthogonal', or 'rotatable'
center : array
    A 1-by-2 array of integers indicating the number of center points
    assigned in each block of the response surface design. Default is
    (1, 1).

Returns
-------
H : 2d-array
    The star-point portion of the design matrix (i.e. at +/- alpha)
a : scalar
    The alpha value to scale the star points with.

Example
-------
::

    >>> star(3)
    array([[-1.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 0., -1.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0., -1.],
            [ 0.,  0.,  1.]])

*/
pub fn star(n: usize, alpha: &str, center: Vec<u32>) -> Result<(Array2<f32>, f32), String> {
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
    use ndarray::{arr2, Zip};

    fn arrays_are_close(a: &Array2<f32>, b: &Array2<f32>, tolerance: f32) -> bool {
        Zip::from(a)
            .and(b)
            .fold(true, |acc, &a, &b| acc && (a - b).abs() <= tolerance)
    }

    #[test]
    fn bbdesign_1() {
        let input = 3;
        let expected = arr2(&[
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
        ]);
        assert_eq!(bbdesign(input), expected);
    }

    #[test]
    fn bbdesign_2() {
        let input = 5;
        let expected = arr2(&[
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
        ]);
        assert_eq!(bbdesign(input), expected);
    }

    #[test]
    fn bbdesign_center_1() {
        let n = 3;
        let center = 1;
        let expected = arr2(&[
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
        ]);
        assert_eq!(bbdesign_center(n, center), expected);
    }

    #[test]
    fn bbdesign_center_2() {
        let n = 4;
        let center = 5;
        let expected = arr2(&[
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
        ]);
        assert_eq!(bbdesign_center(n, center), expected);
    }

    #[test]
    fn ccdesign_o_c() {
        let n = 2;
        let center = vec![2, 2];
        let alpha = "orthogonal";
        let face = "circumscribed";
        let expected: Array2<f32> = arr2(&[
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
        ]);
        let return_array = ccdesign(n, center, alpha, face).unwrap();

        assert!(arrays_are_close(&return_array, &expected, 0.01));
    }

    #[test]
    fn ccdesign_r_c() {
        let n = 3;
        let center = vec![4, 4];
        let alpha = "rotatable";
        let face = "circumscribed";
        let expected: Array2<f32> = arr2(&[
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
        ]);
        let return_array = ccdesign(n, center, alpha, face).unwrap();

        assert!(arrays_are_close(&return_array, &expected, 0.01));
    }

    #[test]
    fn ccdesign_o_i() {
        let n = 4;
        let center = vec![5, 5];
        let alpha = "orthogonal";
        let face = "inscribed";
        let expected: Array2<f32> = arr2(&[
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
        ]);
        let return_array = ccdesign(n, center, alpha, face).unwrap();

        assert!(arrays_are_close(&return_array, &expected, 0.01));
    }

    #[test]
    fn ccdesign_r_i() {
        let n = 2;
        let center = vec![3, 4];
        let alpha = "rotatable";
        let face = "inscribed";
        let expected: Array2<f32> = arr2(&[
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
        ]);
        let return_array = ccdesign(n, center, alpha, face).unwrap();

        assert!(arrays_are_close(&return_array, &expected, 0.01));
    }

    #[test]
    fn ccdesign_f() {
        let n = 3;
        let center = vec![4, 1];
        let alpha = "orthogonal";
        let face = "faced";
        let expected: Array2<f32> = arr2(&[
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
        ]);
        let return_array = ccdesign(n, center, alpha, face).unwrap();

        assert!(arrays_are_close(&return_array, &expected, 0.01));
    }
}
