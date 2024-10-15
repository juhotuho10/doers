# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) but MINOR version are breaking before 1.0.0


## [Unreleased]
### Added
- 

### Fixed
- 

## [0.2.0] - 2024-10-15
### Interface breaking chages:
because: for many functions, the return type is way too high compared to what is possible for it return
also usize isn't really meant to be an argument type, so I'm changing that to u16 

-   `ff2n(n: usize) -> Result<Array2<i32>, String>`

    changed to 

    ```rust
    ff2n(n: u16) Result<Array2<i16>, String>
    ```

-   `gsd(levels: &[u16], reduction: usize, n_designs: usize) -> Result<Vec<Array2<u16>>, String>`

    changed to

    ```rust
    gsd(levels: &[u16], reduction: u16, n_designs: u16) -> Result<Vec<Array2<u16>>, String>
    ```
    
-   `bbdesign(n: usize) -> Result<Array2<i32>, String>`

    changed to 

    ```rust
    bbdesign(n: u16) -> Result<Array2<i16>, String> 
    ```

-   `bbdesign_center(n: usize, center: usize) -> Result<Array2<i32>, String>`

    changed to 

    ```rust
    bbdesign_center(n: u16, center: u16) -> Result<Array2<i16>, String>
    ```

-   `ccdesign(n: usize, center: &[u32], alpha: Alpha, face: Face) -> Result<Array2<f32>, String>`

    changed to 

    ```rust
    ccdesign(n: u16, center: &[u32], alpha: Alpha, face: Face) -> Result<Array2<f32>, String> 
    ```

-   `star(n: usize, alpha: Alpha, center: &[u32]) -> (Array2<f32>, f32)`

    changed to

    ```rust
    star(n: u16, alpha: Alpha, center: &[u32]) -> (Array2<f32>, f32)
    ```

-   `pbdesign(n: u32) -> Array2<i32>`

    changed to

    ```rust
    pbdesign(n: u32) -> Array2<i16>
    ```

-   `fracfact(design: &str) -> Array2<i32>`

    changed to

    ```rust
    fracfact(design: &str) -> Array2<i16>
    ```

-   `lhs_classic(n: usize, samples: usize, random_state: u64) -> Array2<f32>`

    changed to

    ```rust
    lhs_classic(n: u16, samples: u16, random_state: u64) -> Array2<f32>
    ```

-   `lhs_centered(n: usize, samples: usize, random_state: u64) -> Array2<f32>`

    changed to

    ```rust
    lhs_centered(n: u16, samples: u16, random_state: u64) -> Array2<f32>
    ```

-   `lhs_maximin(n: usize, samples: usize, random_state: u64, iterations: u16) -> Array2<f32>`

    changed to

    ```rust
    lhs_maximin(n: u16, samples: u16, random_state: u64, iterations: u16) -> Array2<f32>
    ```

-   `lhs_correlate(n: usize, samples: usize, random_state: u64, iterations: u16) -> Array2<f32>`

    changed to

    ```rust
    lhs_correlate(n: u16, samples: u16, random_state: u64, iterations: u16) -> Array2<f32>
    ```

-   `lhs_mu(n: usize, samples: usize, random_state: u64) -> Array2<f32>`

    changed to 

    ```rust
    lhs_mu(n: u16, samples: u16, random_state: u64) -> Array2<f32>
    ```

### Other changes:
-   fixing documentation



## [0.1.0] - 2024-10-13
### Added
-   Initial release of the crate.