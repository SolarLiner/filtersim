#![allow(dead_code)]
use std::f64::consts::TAU;

use num_traits::Float;
use rustfft::{num_complex::Complex, FftNum, FftPlanner};

use crate::utils::{normalize, zeros};

pub fn sinc<T: Float>(x: T) -> T {
    if x.is_zero() {
        T::one()
    } else {
        x.sin() / x
    }
}

pub fn sinc_filter(fac: usize, beta: f64) -> Vec<f64> {
    let mut kernel = zeros(sinc_filter_len(beta));
    sinc_filter_inplace(&mut kernel, fac);
    kernel
}

pub fn sinc_filter_len(beta: f64) -> usize {
    let len = f64::ceil(4. / 1e-10.max(beta)) as usize;

    if len % 2 == 0 {
        len + 1
    } else {
        len
    }
}

pub fn sinc_filter_inplace(kernel: &mut [f64], fac: usize) {
    let fc = 0.5 / fac as f64;
    let len_f = kernel.len() as f64;
    for (i, s) in kernel.iter_mut().enumerate() {
        let x = i as f64;
        *s = sinc(TAU * fc * (x - (len_f - 1.) / 2.));
    }
    hamming(kernel);
    normalize(kernel);
}

pub fn hamming(window: &mut [f64]) {
    let n = window.len() as f64;
    for (i, s) in window.iter_mut().enumerate() {
        let x = i as f64;
        *s *= 0.54 - 0.46 * f64::cos(TAU * x / n);
    }
}

pub fn fft<T: FftNum>(data: Vec<T>) -> Vec<Complex<T>>
where
    Complex<T>: From<T>,
{
    let mut cbuf = data.into_iter().map(Complex::from).collect::<Vec<_>>();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(cbuf.len());
    fft.process(&mut cbuf);
    cbuf
}

pub fn fft_slice<T: Copy + FftNum>(data: &[T]) -> Vec<Complex<T>> {
    let mut cbuf = data.iter().copied().map(Complex::from).collect::<Vec<_>>();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(cbuf.len());
    fft.process(&mut cbuf);
    cbuf
}

pub fn ifft<T: FftNum>(mut data: Vec<Complex<T>>) -> Vec<T> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(data.len());
    fft.process(&mut data);
    data.into_iter().map(|c| c.re).collect()
}
