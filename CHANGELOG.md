# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- 

### Fixed
- 

## [0.2.0] - 2024-10-15
### Minor interface breaking chages:
because: for many functions, the return type is wayyy to high compared to what is possible for it return
         also usize isn't really meant to be an arguement type, so I'm changing that also to u16 as soon as possible

-   ff2n(n: usize) -> Result<Array2<i32>, String>
    changed to 
    ff2n(n: u16) Result<Array2<i16>, String>

-   gsd(levels: &[u16], reduction: usize, n_designs: usize) -> Result<Vec<Array2<u16>>, String>
    changed to
    gsd(levels: &[u16], reduction: u16, n_designs: u16) -> Result<Vec<Array2<u16>>, String>

-   bbdesign(n: usize) -> Result<Array2<i32>, String> 
    changed to 
    bbdesign(n: u16) -> Result<Array2<i16>, String> 

-   bbdesign_center(n: usize, center: usize) -> Result<Array2<i32>, String>
    changed to 
    bbdesign_center(n: u16, center: usize) -> Result<Array2<i16>, String>

-   ccdesign(n: usize, center: &[u32], alpha: Alpha, face: Face) -> Result<Array2<f32>, String> 
    changed to 
    ccdesign(n: u16, center: &[u32], alpha: Alpha, face: Face) -> Result<Array2<f32>, String> 

-   star(n: usize, alpha: Alpha, center: &[u32]) -> (Array2<f32>, f32)
    changed to
    star(n: u16, alpha: Alpha, center: &[u32]) -> (Array2<f32>, f32)

-   pbdesign(n: u32) -> Array2<i32>
    changed to
    pbdesign(n: u32) -> Array2<i16>

-   fracfact(design: &str) -> Array2<i32>
    changed to
    fracfact(design: &str) -> Array2<i16>

-   lhs_classic(n: usize, samples: usize, random_state: u64) -> Array2<f32>
    changed to
    lhs_classic(n: u16, samples: u16, random_state: u64) -> Array2<f32>

-   lhs_centered(n: usize, samples: usize, random_state: u64) -> Array2<f32>
    changed to
    lhs_centered(n: u16, samples: u16, random_state: u64) -> Array2<f32>

-   lhs_maximin(n: usize, samples: usize, random_state: u64, iterations: u16) -> Array2<f32>
    changed to
    lhs_maximin(n: u16, samples: u16, random_state: u64, iterations: u16) -> Array2<f32>

-   lhs_correlate(n: usize, samples: usize, random_state: u64, iterations: u16) -> Array2<f32>
    changed to
    lhs_correlate(n: u16, samples: u16, random_state: u64, iterations: u16) -> Array2<f32>

-   lhs_mu(n: usize, samples: usize, random_state: u64) -> Array2<f32>
    changed to 
    lhs_mu(n: u16, samples: u16, random_state: u64) -> Array2<f32>


### Other changes:
-   fixing documentation



## [0.1.0] - 2024-10-13
### Added
-   Initial release of the crate.