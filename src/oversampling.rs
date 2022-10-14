use nih_plug::nih_dbg;
use num_traits::Zero;

use crate::fft_convolve::FftConvolve;
use crate::math::{sinc_filter_inplace, sinc_filter_len};
use crate::overlap_add::OverlapAdd;
use crate::utils::zeros;

#[derive(Debug, Clone)]
pub struct Oversample {
    fac: usize,
    os_buffer: Vec<f64>,
    filter: FftConvolve<f64>,
    oadd: OverlapAdd<f64>,
}

impl Oversample {
    pub fn new(fac: usize, max_block_size: usize) -> Self {
        let os_size = max_block_size * fac;
        let filter_len = nih_dbg!(sinc_filter_len(8e-2));
        let max_len = os_size + filter_len - 1;

        let mut filter = zeros(max_len);
        sinc_filter_inplace(&mut filter[..filter_len], fac);
        let filter = FftConvolve::new(&filter);
        Self {
            fac,
            os_buffer: zeros(os_size),
            filter,
            oadd: OverlapAdd::new(os_size, filter_len - 1),
        }
    }

    pub fn latency_samples(&self) -> u32 {
        self.oadd.padding() as u32 / 4
    }

    pub fn max_buffer_size(&self) -> usize {
        self.os_buffer.len() / self.fac
    }

    pub fn with_oversample<R>(
        &mut self,
        input: &mut [f64],
        mut f: impl FnMut(&mut [f64]) -> R,
    ) -> R {
        self.zero_stuff(input);
        let os_len = input.len() * self.fac;
        let res = self.oadd.process(&mut self.os_buffer[..os_len], |input| {
            self.filter.process(input);
            // let res = f(&mut input[..os_len]);
            let res = f(input);
            self.filter.process(input);
            res
        });
        self.decimate(input);
        res
    }

    pub fn reset(&mut self) {
        self.oadd.reset();
    }

    fn zero_stuff(&mut self, input: &mut [f64]) {
        assert!(input.len() <= self.max_buffer_size());

        for (i, s) in input.iter().copied().enumerate() {
            self.os_buffer[self.fac * i] = self.fac as f64 * s;
            for j in 1..self.fac {
                self.os_buffer[self.fac * i + j].set_zero();
            }
        }
    }

    fn decimate(&mut self, dest: &mut [f64]) {
        assert!(dest.len() <= self.max_buffer_size());

        for (i, d) in dest.iter_mut().enumerate() {
            *d = self.os_buffer[i * self.fac];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Oversample;

    #[test]
    fn test_null() {
        let mut oversample = Oversample::new(4, 16);
        let input = [-1.0, 1.0, 1.0, 0.0];
        let mut buffer = input;
        let mut buffer2 = [0.0; 10];
        // let expected = buffer;
        oversample.with_oversample(&mut buffer, |arr| {
            assert_eq!(arr.len(), 4 * 4);
        });
        oversample.with_oversample(&mut buffer2, |arr| {
            assert!(arr.iter().any(|x| *x > 0.));
        });

        eprintln!("{:?}", buffer);
        eprintln!("{:?}", buffer2);
        assert_eq!(input, buffer);
    }

    #[cfg(never)]
    #[test]
    fn test_oversample() {
        let mut oversample = Oversample::new(4, 16);
        let mut buffer = [0.0; 16];
        oversample.with_oversample(&mut buffer, |arr| {
            let step = 1.0 / 5.0;
            for (i, s) in arr.iter_mut().enumerate() {
                *s = 2. * f64::fract((i as f64) / step) - 1.0;
            }
        });

        eprintln!("{:?}", buffer);
        todo!();
    }
}
