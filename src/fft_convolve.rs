use std::cmp::Ordering;
use std::{fmt, ops, sync::Arc};

use nih_plug::nih_error;
use num_traits::{FromPrimitive, NumAssign};
use rustfft::{num_complex::Complex, Fft, FftNum, FftPlanner};

use crate::math::fft_slice;
use crate::utils::{zero_out, zeros};

#[derive(Clone)]
pub struct FftConvolve<T> {
    fft_kernel: Vec<Complex<T>>,
    fft_buffer: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
    fwd: Arc<dyn Fft<T>>,
    inv: Arc<dyn Fft<T>>,
}

impl<T> fmt::Debug for FftConvolve<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(std::any::type_name::<Self>())
            .field("fft_kernel", &format!("[.. len {}]", self.fft_kernel.len()))
            .field("fft_buffer", &format!("[.. len {}]", self.fft_buffer.len()))
            .finish_non_exhaustive()
    }
}

impl<T> FftConvolve<T> {
    pub fn buffer_size(&self) -> usize {
        self.fft_buffer.len()
    }
}

impl<T: Copy + FftNum> FftConvolve<T> {
    /// Create a convolution operator from an already transformed kernel.
    ///
    /// The kernel needs to already have been padded with zeros to match the output length,
    /// that is (N + K - 1) where the input is size N and kernel size K.
    /// The padding needs to be added before the Fourier Transform to ensure correct interpretation.
    pub fn new_from_fft(fft_kernel: impl Into<Vec<Complex<T>>>) -> Self {
        let fft_kernel = fft_kernel.into();
        let fft_buffer = zeros(fft_kernel.len());
        let mut planner = FftPlanner::new();
        let fwd = planner.plan_fft_forward(fft_kernel.len());
        let inv = planner.plan_fft_inverse(fft_kernel.len());
        let scratch = zeros(
            fwd.get_inplace_scratch_len()
                .max(inv.get_inplace_scratch_len()),
        );
        Self {
            fft_kernel,
            fft_buffer,
            scratch,
            fwd,
            inv,
        }
    }

    /// Create a convolution operator from a provided kernel.
    ///
    /// The kernel needs to already have been padded with zeros to match the output length,
    /// that is (N + K - 1) where the input is size N and kernel size K.
    pub fn new(kernel: &[T]) -> Self {
        Self::new_from_fft(fft_slice(kernel))
    }
}

impl<T: Copy + NumAssign + FftNum + FromPrimitive> FftConvolve<T> {
    pub fn process(&mut self, input: &mut [T]) {
        for (c, s) in self.fft_buffer.iter_mut().zip(input.iter().copied()) {
            *c = Complex::from(s);
        }

        nih_plug::util::permit_alloc(|| match input.len().cmp(&self.buffer_size()) {
            Ordering::Less => {
                nih_error!("Input buffer is smaller than expected; padding with zeros (can cause audible artifacts): expected {} but found {}", self.buffer_size(), input.len());
                zero_out(&mut self.fft_buffer[input.len()..]);
            }
            Ordering::Greater => {
                nih_error!(
                    "Input buffer is bigger than expected - this *will* result in audible artifacts: expected {}, found: {}", self.buffer_size(), input.len()
                );
            }
            _ => {}
        });

        self.fwd
            .process_with_scratch(&mut self.fft_buffer, &mut self.scratch);
        mul_inplace(&mut self.fft_buffer, &self.fft_kernel);
        self.inv
            .process_with_scratch(&mut self.fft_buffer, &mut self.scratch);

        let len_t = T::from_usize(input.len()).unwrap();
        for (s, c) in input.iter_mut().zip(self.fft_buffer.iter()) {
            *s = c.re / len_t;
        }
    }
}

fn mul_inplace<T: Copy + ops::MulAssign>(dest: &mut [T], factors: &[T]) {
    assert_eq!(dest.len(), factors.len());
    for (d, s) in dest.iter_mut().zip(factors.iter().copied()) {
        *d *= s;
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::FftConvolve;

    #[test]
    fn simple_convolve() {
        let input = [
            1. / 7.,
            1. / 7.,
            1. / 7.,
            1. / 7.,
            1. / 7.,
            1. / 7.,
            1. / 7.,
            0.,
            0.,
            0.,
        ];
        let kernel = [1. / 4., 1. / 4., 1. / 4., 1. / 4., 0., 0., 0., 0., 0., 0.];
        let mut filter = FftConvolve::new(&kernel);
        let mut output = input;
        filter.process(&mut output);
        println!("{:?}", filter.fft_kernel);
        println!("{:?}", output);
        let energy = output.into_iter().sum::<f64>();
        assert_abs_diff_eq!(energy, 1.0);
    }
}
