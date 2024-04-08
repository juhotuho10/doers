use ndarray_rand::rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use ndarray_rand::rand_distr::{Distribution, Uniform}; // For uniform distribution
use ndarray_rand::RandomExt; // For random array generation // For seeding the random number generator // A small, fast RNG

use ndarray::stack;
use ndarray::{s, Array, Array1, Array2, ArrayBase, Axis, Data, Ix2, Zip};
use ndarray_rand::rand::{thread_rng, Rng};
use std::vec;

/*
Generate a latin-hypercube design

Parameters
----------
n : int
    The number of factors to generate samples for

Optional
--------
samples : int
    The number of samples to generate for each factor (Default: n)
criterion : str
    Allowable values are "center" or "c", "maximin" or "m",
    "centermaximin" or "cm", and "correlation" or "corr". If no value
    given, the design is simply randomized.
iterations : int
    The number of iterations in the maximin and correlations algorithms
    (Default: 5).
randomstate : np.random.RandomState, int
     Random state (or seed-number) which controls the seed and random draws
correlation_matrix : ndarray
     Enforce correlation between factors (only used in lhs_mu)

Returns
-------
H : 2d-array
    An n-by-samples design matrix that has been normalized so factor values
    are uniformly spaced between zero and one.

Example
-------
A 3-factor design (defaults to 3 samples)::

    >>> lhs(3, random_state=42)
    array([[ 0.12484671,  0.95539205,  0.24399798],
           [ 0.53288616,  0.38533955,  0.86703834],
           [ 0.68602787,  0.31690477,  0.38533151]])

A 4-factor design with 6 samples::

    >>> lhs(4, samples=6, random_state=42)
    array([[ 0.06242335,  0.19266575,  0.88202411,  0.89439364],
           [ 0.19266977,  0.53538985,  0.53030416,  0.49498498],
           [ 0.71737371,  0.75412607,  0.17634727,  0.71520486],
           [ 0.63874044,  0.85658231,  0.33676408,  0.31102936],
           [ 0.43351917,  0.45134543,  0.12199899,  0.53056742],
           [ 0.93530882,  0.15845238,  0.7386575 ,  0.09977641]])

A 2-factor design with 5 centered samples::

    >>> lhs(2, samples=5, criterion='center', random_state=42)
    array([[ 0.1,  0.9],
           [ 0.5,  0.5],
           [ 0.7,  0.1],
           [ 0.3,  0.7],
           [ 0.9,  0.3]])

A 3-factor design with 4 samples where the minimum distance between
all samples has been maximized::

    >>> lhs(3, samples=4, criterion='maximin', random_state=42)
    array([[ 0.69754389,  0.2997106 ,  0.96250964],
           [ 0.10585037,  0.09872038,  0.73157522],
           [ 0.25351996,  0.65148999,  0.07337204],
           [ 0.91276926,  0.97873992,  0.42783549]])

A 4-factor design with 5 samples where the samples are as uncorrelated
as possible (within 10 iterations)::

    >>> lhs(4, samples=5, criterion='correlation', iterations=10, random_state=42)
    array([[ 0.72088348,  0.05121366,  0.97609357,  0.92487081],
           [ 0.49507404,  0.51265511,  0.00808672,  0.37915272],
           [ 0.22217816,  0.2878673 ,  0.24034384,  0.42786629],
           [ 0.91977309,  0.93895699,  0.64061224,  0.14213258],
           [ 0.04719698,  0.70796822,  0.53910322,  0.78857071]])

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

#[allow(dead_code)]
pub fn lhs_maximin(
    n: usize,
    samples: usize,
    random_state: u64,
    iterations: u16,
    centered: bool,
) -> Array2<f32> {
    let mut max_dist = 0.;
    let mut h_array: Array2<f32> = Array2::from_elem((n, samples), 0.);
    let mut rng = SmallRng::seed_from_u64(random_state);

    for _ in 0..iterations {
        let random_int = rng.gen_range(0..u64::MAX);
        rng = SmallRng::seed_from_u64(random_int);
        // Assuming lhs_classic and lhs_centered are modified to accept &mut rng instead of random_state
        let h_candidate = match centered {
            false => lhs_classic(n, samples, random_int),
            true => lhs_centered(n, samples, random_int),
        };

        let dist_array = pairwise_euclidean_dist(&h_candidate).to_owned();

        // Assuming implementation for pairwise_euclidean_dist provided elsewhere
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
#[allow(dead_code)]
pub fn lhs_correlate(n: usize, samples: usize, random_state: u64, iterations: u16) -> Array2<f32> {
    let mut mincorr: f32 = f32::INFINITY;
    let mut h_array = Array2::<f32>::zeros((samples, n));
    let mut rng = SmallRng::seed_from_u64(random_state);

    for _ in 0..iterations {
        let random_int = rng.gen_range(0..u64::MAX);
        rng = SmallRng::seed_from_u64(random_int);
        let h_candidate = lhs_classic(n, samples, random_int);
        let corr = corrcoef(&h_array.t());
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

        let min_l = argmin_ignore_nan(&avg_dist);

        match min_l {
            Some(min_l) => {
                for j in 0..d_ij.ncols() {
                    d_ij[[min_l, j]] = f32::NAN;
                    d_ij[[j, min_l]] = f32::NAN;
                }

                index_rm[i] = min_l;
            }
            None => panic!(),
        }
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

fn euclidean_distance(a: &Array1<f32>, b: Array1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Computes the pairwise distances between rows in a 2D array.
fn pairwise_euclidean_dist(input: &Array2<f32>) -> Array1<f32> {
    let rows = input.nrows();
    let mut distances = Vec::new();

    for i in 0..rows {
        for j in i + 1..rows {
            let distance = euclidean_distance(&input.row(i).to_owned(), input.row(j).to_owned());

            distances.push(distance);
        }
    }

    Array1::from(distances)
}
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

fn corrcoef<S>(x: &ArrayBase<S, Ix2>) -> Array2<f32>
where
    S: Data<Elem = f32>,
{
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

fn max_abs_off_diagonal(r: &Array2<f32>) -> f32 {
    let identity: Array2<f32> = Array2::eye(r.nrows());
    let abs_diff = (r - identity).mapv_into(f32::abs);
    let max_abs_off_diag = abs_diff.iter().fold(0.0_f32, |acc, &x| acc.max(x));

    max_abs_off_diag
}
fn mean_of_first_two(values: &[f32]) -> f32 {
    let valid_values: Vec<f32> = values.iter().filter(|&&v| !v.is_nan()).cloned().collect();
    let total: f32 = valid_values.iter().take(2).sum();
    total / valid_values.len() as f32
}

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

fn sort_array2_by_axis_with_nan_handling(mut array: Array2<f32>) -> Array2<f32> {
    // Iterate over each row
    for mut row in array.axis_iter_mut(Axis(0)) {
        // Convert the row to a Vec<f32>, sort it, then update the row
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
    use ndarray::{Array3, Zip};

    fn sort_ndarray_array1(array: Array1<f32>) -> Array1<f32> {
        let mut vec = array.to_vec();
        vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Array1::from(vec)
    }

    fn arrays2_are_close(a: &Array2<f32>, b: &Array2<f32>, tolerance: f32) -> bool {
        Zip::from(a)
            .and(b)
            .fold(true, |acc, &a, &b| acc && (a - b).abs() <= tolerance)
    }

    fn arrays1_are_close(a: &Array1<f32>, b: &Array1<f32>, tolerance: f32) -> bool {
        Zip::from(a)
            .and(b)
            .fold(true, |acc, &a, &b| acc && (a - b).abs() <= tolerance)
    }

    fn vec_array2_to_array3(arrays: Vec<Array2<f32>>) -> Array3<f32> {
        let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
        stack(Axis(0), &views).expect("Error stacking arrays")
    }

    #[test]
    fn lhs_classic_1() {
        let n = 1;
        let samples = 1;
        let random_state = 42;

        lhs_classic(n, samples, random_state);
    }

    #[test]
    fn lhs_classic_2() {
        let n = 13;
        let samples = 12;

        let mut vectors: Vec<Array2<f32>> = vec![];
        for i in 0..1000 {
            vectors.push(lhs_classic(n, samples, i));
        }

        let collection_array = vec_array2_to_array3(vectors);
        let avg_array = collection_array.mean_axis(Axis(0)).unwrap();

        let center = Array2::from_elem((samples, n), 0.5);

        assert!(arrays2_are_close(&center, &avg_array.to_owned(), 0.05));
    }

    #[test]
    fn lhs_classic_3() {
        let n = 6;
        let samples = 6;

        let mut vectors: Vec<Array2<f32>> = vec![];
        for i in 0..1000 {
            vectors.push(lhs_classic(n, samples, i));
        }

        let collection_array = vec_array2_to_array3(vectors);
        let avg_array = collection_array.mean_axis(Axis(0)).unwrap();

        let center = Array2::from_elem((samples, n), 0.5);

        assert!(arrays2_are_close(&center, &avg_array.to_owned(), 0.05));
    }

    #[test]
    fn lhs_centered_1() {
        let n = 2;
        let samples = 3;

        let mut vectors: Vec<Array2<f32>> = vec![];
        for i in 0..5000 {
            vectors.push(lhs_centered(n, samples, i));
        }

        let collection_array = vec_array2_to_array3(vectors);
        let avg_array = collection_array.mean_axis(Axis(0)).unwrap();

        let center = Array2::from_elem((samples, n), 0.5);

        assert!(arrays2_are_close(&center, &avg_array.to_owned(), 0.05));
    }

    #[test]
    fn lhs_centered_2() {
        let n = 6;
        let samples = 20;
        let rand = 1;

        let arr = lhs_centered(n, samples, rand);

        let mut cut: Array1<f64> = Array::linspace(0., 1., samples + 1);
        cut = cut.mapv(|x: f64| (x * 100.).round() / 100.); // rounding the variables to 0.01

        let a = cut.slice(s![..samples]);
        let b = cut.slice(s![1..samples + 1]);
        let mut center = ((&a + &b) / 2.).mapv(|x| x as f32);
        center = sort_ndarray_array1(center);

        for col in arr.axis_iter(Axis(1)) {
            assert_eq!(center, sort_ndarray_array1(col.to_owned()))
        }
    }

    #[test]
    fn lhs_maximin_classic_1() {
        let n = 2;
        let samples = 2;
        let iterations = 4;
        let centered = false;

        let mut vectors: Vec<Array2<f32>> = vec![];
        for i in 0..5000 {
            vectors.push(lhs_maximin(n, samples, i, iterations, centered));
        }

        let collection_array = vec_array2_to_array3(vectors);
        let avg_array = collection_array.mean_axis(Axis(0)).unwrap();

        let center = Array2::from_elem((samples, n), 0.5);

        assert!(arrays2_are_close(&center, &avg_array.to_owned(), 0.05));
    }

    #[test]
    fn lhs_maximin_classic_2() {
        let n = 10;
        let samples = 15;
        let iterations = 4;
        let centered = false;

        let mut vectors: Vec<Array2<f32>> = vec![];
        for i in 0..5000 {
            vectors.push(lhs_maximin(n, samples, i, iterations, centered));
        }

        let collection_array = vec_array2_to_array3(vectors);
        let avg_array = collection_array.mean_axis(Axis(0)).unwrap();

        let center = Array2::from_elem((samples, n), 0.5);

        assert!(arrays2_are_close(&center, &avg_array.to_owned(), 0.05));
    }

    #[test]
    fn lhs_maximin_centered_1() {
        let n = 2;
        let samples = 2;
        let iterations = 10;
        let centered = true;

        let mut vectors: Vec<Array2<f32>> = vec![];
        for i in 0..10000 {
            vectors.push(lhs_maximin(n, samples, i, iterations, centered));
        }

        let collection_array = vec_array2_to_array3(vectors);
        let avg_array = collection_array.mean_axis(Axis(0)).unwrap();

        let center = Array2::from_elem((samples, n), 0.5);

        assert!(arrays2_are_close(&center, &avg_array.to_owned(), 0.05));
    }

    #[test]
    fn lhs_maximin_centered_2() {
        let n = 14;
        let samples = 11;
        let iterations = 4;
        let centered = true;

        let arr = lhs_maximin(n, samples, 42, iterations, centered);

        let mut cut: Array1<f64> = Array::linspace(0., 1., samples + 1);
        cut = cut.mapv(|x: f64| (x * 100.).round() / 100.); // rounding the variables to 0.01

        let a = cut.slice(s![..samples]);
        let b = cut.slice(s![1..samples + 1]);
        let mut center = ((&a + &b) / 2.).mapv(|x| x as f32);
        center = sort_ndarray_array1(center);

        for col in arr.axis_iter(Axis(1)) {
            assert_eq!(center, sort_ndarray_array1(col.to_owned()))
        }
    }

    #[test]
    fn lhs_correlate_1() {
        let n = 2;
        let samples = 2;
        let iterations = 10;

        let mut vectors: Vec<Array2<f32>> = vec![];
        for i in 0..5000 {
            vectors.push(lhs_correlate(n, samples, i, iterations));
        }

        let collection_array = vec_array2_to_array3(vectors);
        let avg_array = collection_array.mean_axis(Axis(0)).unwrap();

        let center = Array2::from_elem((samples, n), 0.5);

        assert!(arrays2_are_close(&center, &avg_array.to_owned(), 0.05));
    }

    #[test]
    fn lhs_correlate_2() {
        let n = 13;
        let samples = 9;
        let iterations = 4;

        let mut vectors: Vec<Array2<f32>> = vec![];
        for i in 0..5000 {
            vectors.push(lhs_correlate(n, samples, i, iterations));
        }

        let collection_array = vec_array2_to_array3(vectors);
        let avg_array = collection_array.mean_axis(Axis(0)).unwrap();

        let center = Array2::from_elem((samples, n), 0.5);

        assert!(arrays2_are_close(&center, &avg_array.to_owned(), 0.05));
    }

    #[test]
    fn lhs_mu_1() {
        let n = 5;
        let samples = 6;

        let mut vectors: Vec<Array2<f32>> = vec![];
        for i in 0..5000 {
            vectors.push(lhs_mu(n, samples, i));
        }

        let collection_array = vec_array2_to_array3(vectors);
        let avg_array = collection_array.mean_axis(Axis(0)).unwrap();

        let center = Array2::from_elem((samples, n), 0.5);

        assert!(arrays2_are_close(&center, &avg_array.to_owned(), 0.1));
    }

    #[test]
    fn lhs_mu_2() {
        let n = 5;
        let samples = 5;

        let arr = lhs_mu(n, samples, 42);

        let mut cut: Array1<f64> = Array::linspace(0., 1., samples + 1);
        cut = cut.mapv(|x: f64| (x * 100.).round() / 100.); // rounding the variables to 0.01

        let a = cut.slice(s![..samples]);
        let b = cut.slice(s![1..samples + 1]);
        let mut center = ((&a + &b) / 2.).mapv(|x| x as f32);
        center = sort_ndarray_array1(center);

        for col in arr.axis_iter(Axis(1)) {
            let sorted_col = sort_ndarray_array1(col.to_owned());
            assert!(arrays1_are_close(
                &center,
                &sorted_col,
                0.5 / (samples as f32)
            ));
        }
    }

    #[test]
    fn lhs_mu_3() {
        let n = 17;
        let samples = 15;

        let arr = lhs_mu(n, samples, 42);

        let mut cut: Array1<f64> = Array::linspace(0., 1., samples + 1);
        cut = cut.mapv(|x: f64| (x * 100.).round() / 100.); // rounding the variables to 0.01

        let a = cut.slice(s![..samples]);
        let b = cut.slice(s![1..samples + 1]);
        let mut center = ((&a + &b) / 2.).mapv(|x| x as f32);
        center = sort_ndarray_array1(center);

        for col in arr.axis_iter(Axis(1)) {
            let sorted_col = sort_ndarray_array1(col.to_owned());
            assert!(arrays1_are_close(
                &center,
                &sorted_col,
                0.5 / (samples as f32)
            ));
        }
    }
}
