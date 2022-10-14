#![allow(dead_code)]
use std::f64::{consts::TAU, EPSILON};

use num_traits::Float;

#[derive(Debug, Clone, Copy)]
pub struct RcFilter {
    rc: f64,
}

impl RcFilter {
    pub fn new(fc: f64) -> Self {
        Self {
            rc: (TAU * fc).recip(),
        }
    }

    pub fn set_fc(&mut self, fc: f64) {
        self.rc = (TAU * fc).recip();
    }

    pub fn dv(&self, state: RcFilterState) -> f64 {
        (state.v_in - state.v_c) / self.rc
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RcFilterState {
    pub v_in: f64,
    pub v_c: f64,
}

impl Default for RcFilterState {
    fn default() -> Self {
        Self { v_c: 0., v_in: 0. }
    }
}

impl RcFilterState {
    pub fn process(&mut self, filter: &RcFilter, step: f64) -> f64 {
        self.v_c += filter.dv(*self) * step;
        self.v_c
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ActiveLpf {
    pub rc: RcFilter,
    pub vcc: f64,
    pub amp: f64,
}

impl ActiveLpf {
    pub fn new(fc: f64) -> Self {
        Self {
            amp: 1.,
            vcc: 12.,
            rc: RcFilter::new(fc),
        }
    }

    pub fn set_fc(&mut self, fc: f64) {
        self.rc.set_fc(fc);
    }

    pub fn set_amp(&mut self, amp: f64) {
        self.amp = amp;
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ActiveLpfState {
    rc: RcFilterState,
}

impl ActiveLpfState {
    pub fn set_v_in(&mut self, v_in: f64) {
        self.rc.v_in = v_in;
    }
}

impl ActiveLpfState {
    pub fn process(&mut self, filter: &ActiveLpf, step: f64) -> f64 {
        self.rc.process(&filter.rc, step);
        clamp(
            -filter.vcc,
            filter.vcc,
            filter.amp / (1. + filter.amp) * self.rc.v_c,
        )
    }
}

fn clamp<T: Float>(a: T, b: T, x: T) -> T {
    x.min(b).max(a)
}

/* fn dclamp<T: Float>(a: T, b: T, x: T) -> T {
    if x > a && x < b {
        T::one()
    } else {
        T::zero()
    }
} */

pub struct SallenKey {
    rc: f64,
    k: f64,
}

impl SallenKey {
    pub fn new(fc: f64, q: f64) -> Self {
        Self {
            rc: Self::get_rc(fc),
            k: Self::get_k(q),
        }
    }

    pub fn set_fc(&mut self, fc: f64) {
        self.rc = Self::get_rc(fc);
    }

    pub fn set_q(&mut self, q: f64) {
        self.k = Self::get_k(q);
    }

    fn get_rc(fc: f64) -> f64 {
        (TAU * fc).recip()
    }

    fn get_k(q: f64) -> f64 {
        3. - q.max(EPSILON).recip()
    }
}

pub struct SallenKeyState {
    pub v_in: f64,
    last_v_c: f64,
    v_c: f64,
}

impl SallenKeyState {
    pub fn process(&mut self, filter: &SallenKey, step: f64) -> f64 {
        let dv_c = step * (self.last_v_c - self.v_c);
        let den = filter.rc * filter.rc * dv_c / self.v_c / self.v_c
            + filter.rc * (3. + filter.k) * dv_c / self.v_c
            + 1.;
        filter.k / den
    }
}
