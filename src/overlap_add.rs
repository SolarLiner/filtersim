use std::ops;

use nih_plug::nih_error;
use num_traits::Zero;

use crate::utils::{zero_out, zeros};

#[derive(Debug, Clone)]
pub struct OverlapAdd<T> {
    padding: Vec<T>,
    inner_block: Vec<T>,
    max_block_size: usize,
}

impl<T: Zero> OverlapAdd<T> {
    pub fn new(max_block_size: usize, padding: usize) -> Self {
        Self {
            max_block_size,
            padding: zeros(padding),
            inner_block: zeros(max_block_size + padding),
        }
    }

    pub fn padding(&self) -> usize {
        self.padding.len()
    }

    pub fn reset(&mut self) {
        zero_out(&mut self.padding);
        zero_out(&mut self.inner_block);
    }

    #[inline(never)]
    pub fn process<R>(
        &mut self,
        buffer: &mut [T],
        mut process_inner: impl FnMut(&mut [T]) -> R,
    ) -> R
    where
        T: ops::AddAssign + Copy,
    {
        let buffer = if buffer.len() > self.max_block_size {
            nih_error!("Buffer length is higher than configured inner block length. We are truncating the buffer length, and this will result in artifacts");
            &mut buffer[..self.max_block_size]
        } else {
            buffer
        };

        // Initialization (copying into buffers + adding padding)
        let l = buffer.len() + self.padding();
        let inner_buffer = &mut self.inner_block[..l];
        inner_buffer[..buffer.len()].copy_from_slice(buffer);
        for (b, p) in inner_buffer.iter_mut().zip(self.padding.iter().copied()) {
            *b += p;
        }

        let res = process_inner(&mut inner_buffer[..l]);

        // Termination (copying back into user buffer)
        let (to_buffer, to_padding) = inner_buffer.split_at(buffer.len());
        buffer.copy_from_slice(to_buffer);
        self.padding[..to_padding.len()].copy_from_slice(to_padding);
        zero_out(&mut self.padding[to_padding.len()..]);

        res
    }
}
