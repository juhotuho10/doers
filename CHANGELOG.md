# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- 

### Fixed
- 

## [0.2.0] - 2024-10-15
### Interface breaking chages:
-   ff2n(n: usize) -> Result<Array2<i32>, String>
    changed to 
    ff2n(n: u16) Result<Array2<i16>, String> 
    because: to be similar with fullfact interface, also having it be i32 when it can only generate [-1 to 1] values is a little wasteful

-   bbdesign(n: usize) -> Result<Array2<i32>, String> 
    changed to 
    bbdesign(n: u16) -> Result<Array2<i16>, String> 
    for similar resons as listed above

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

### Other changes:
-   fixing documentation



## [0.1.0] - 2024-10-13
### Added
-   Initial release of the crate.