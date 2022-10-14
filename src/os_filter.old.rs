use std::{fmt::Debug, sync::Arc};

use nih_plug::nih_debug_assert_eq;
use num_traits::{Float, FloatConst, FromPrimitive, Num, NumAssign, Signed, Zero};
use rustfft::{num_complex::Complex, Fft, FftPlanner};

pub fn blackman<T: NumAssign + Float + FloatConst>(window: &mut [T]) {
    let size = window.len();

    let scale_1 = T::TAU() / T::from(size - 1).unwrap();
    let scale_2 = scale_1 * T::from(2).unwrap();
    for (i, sample) in window.iter_mut().enumerate() {
        let cos_1 = (scale_1 * T::from(i).unwrap()).cos();
        let cos_2 = (scale_2 * T::from(i).unwrap()).cos();
        *sample *= T::from(0.42).unwrap() - (T::from(0.5).unwrap() * cos_1)
            + (T::from(0.08).unwrap() * cos_2);
    }
}

pub fn bandlimit_sinc<T: NumAssign + Float + FloatConst>(reduction: T, output: &mut [T]) {
    let fc2 = reduction.recip();
    let n = T::from(output.len()).unwrap();
    for (i, s) in output.iter_mut().enumerate() {
        let x = T::from(i).unwrap();
        let x = fc2 * (x - (n - T::one()) / T::from(2).unwrap());
        *s = if x.abs() < T::epsilon() {
            T::one()
        } else {
            x.sin() / x
        };
        *s *= fc2;
    }
    blackman(output);
}

pub fn expand<T: Copy + Num>(x: &mut [T], fac: usize) {
    if fac <= 1 {
        return;
    }
    let inner_size = x.len() / fac;
    for i in (0..inner_size).rev() {
        x[i * fac] = x[i];
        for j in 1..fac {
            x[i * fac + j] = T::zero();
        }
    }
}

pub struct Upsample<T: 'static + Debug + Signed + Float + FromPrimitive + Send + Sync> {
    fac: usize,
    winsize: usize,
    window_fft: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
    fft_scratch: Vec<Complex<T>>,
    fwd: Arc<dyn Fft<T>>,
    inv: Arc<dyn Fft<T>>,
}

impl<
        T: 'static + Debug + Signed + Float + FromPrimitive + Send + Sync + NumAssign + FloatConst,
    > Upsample<T>
{
    pub fn new(fac: usize, window_size: usize) -> Self {
        const B: f64 = 0.08;
        let winsize = ((4.0 / B).ceil() as usize).next_power_of_two();
        let convolution_size = fac * (window_size + winsize - 1);
        let mut planner = FftPlanner::new();
        let fwd = planner.plan_fft_forward(convolution_size);
        let inv = planner.plan_fft_inverse(convolution_size);
        let mut window = (0..convolution_size).map(|_| T::one()).collect::<Vec<_>>();

        bandlimit_sinc(T::from(fac).unwrap(), &mut window);

        let mut window_fft = window.into_iter().map(Complex::from).collect::<Vec<_>>();
        fwd.process(&mut window_fft);

        Self {
            winsize,
            window_fft,
            fac,
            scratch: (0..convolution_size).map(|_| Complex::zero()).collect(),
            fft_scratch: (0..fwd
                .get_inplace_scratch_len()
                .max(inv.get_inplace_scratch_len()))
                .map(|_| Complex::zero())
                .collect(),
            fwd,
            inv,
        }
    }

    pub fn window_size(&self) -> usize {
        self.winsize
    }

    pub fn convolution_size(&self) -> usize {
        self.scratch.len()
    }

    pub fn oversample(&mut self, output: &mut [T]) {
        nih_debug_assert_eq!(output.len(), self.window_fft.len());
        expand(output, self.fac);

        for (o, i) in self.scratch.iter_mut().zip(output.iter()) {
            o.re = *i;
            o.im = T::zero();
        }
        self.fwd
            .process_with_scratch(&mut self.scratch, &mut self.fft_scratch);
        for (i, out) in self.window_fft.iter().copied().zip(self.scratch.iter_mut()) {
            *out *= i;
        }
        self.inv
            .process_with_scratch(&mut self.scratch, &mut self.fft_scratch);
        let gain_reduction = T::from(self.window_fft.len()).unwrap();
        for (out, inp) in output.iter_mut().zip(self.scratch.iter()) {
            *out = inp.re / gain_reduction;
        }
    }

    pub fn downsample(&self, buffer: &mut [T]) {
        if self.fac <= 1 {
            return;
        }
        let inner_len = buffer.len() / self.fac;
        for i in (1..inner_len) {
            buffer[i] = buffer[i * self.fac];
        }
    }
}
