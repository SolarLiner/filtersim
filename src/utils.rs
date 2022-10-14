use std::iter::Sum;

use num_traits::{Float, NumAssign, Zero};

pub fn zeros<T: Zero>(len: usize) -> Vec<T> {
    std::iter::repeat_with(T::zero).take(len).collect()
}

pub fn zero_out<T: Zero>(array: &mut [T]) {
    for s in array {
        s.set_zero();
    }
}

pub fn lerp_buf<T: Float>(output: &mut [T], input: &[T]) {
    let factor = T::from(output.len() - 1).unwrap() / T::from(input.len() - 1).unwrap();
    let step = factor.recip();

    for (i, s) in output.iter_mut().enumerate() {
        let t = T::from(i).unwrap() * step;
        let ti = t.floor().to_usize().unwrap().min(input.len() - 1);
        let tf = t.fract();
        let a = input[ti];
        let b = input[(ti + 1).min(input.len() - 1)];
        *s = (T::one() - tf) * a + tf * b;
    }
}

pub fn normalize<T: Copy + Sum<T> + NumAssign>(normalize: &mut [T]) {
    let sum = normalize.iter().copied().sum::<T>();
    for s in normalize {
        *s /= sum;
    }
}

#[cfg(test)]
mod tests {
    use num_traits::Zero;

    use super::{lerp_buf, zero_out, zeros};

    #[test]
    fn test_zero_out() {
        let mut data = [1; 16];
        zero_out(&mut data);
        assert!(data.iter().all(|x| x.is_zero()));
    }

    #[test]
    fn test_zeros() {
        let data = zeros::<f32>(16);
        assert_eq!(data.len(), 16);
        assert!(data.iter().all(|x| x.is_zero()));
    }

    #[test]
    fn test_lerp_buf() {
        let input = [0.0, 1.0, 2.0];
        let expected = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0];
        let mut actual = zeros(9);
        lerp_buf(&mut actual, &input);
        assert_eq!(actual, expected);
    }
}
