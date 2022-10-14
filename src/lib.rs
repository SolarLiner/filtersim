#![allow(clippy::needless_range_loop)]
mod fft_convolve;
mod lpf;
mod math;
mod overlap_add;
mod oversampling;
mod utils;


use lpf::{ActiveLpf, ActiveLpfState};

use nih_plug::{prelude::*};
use oversampling::Oversample;
use std::{sync::Arc};


struct Filtersim<const CHANNELS: usize> {
    params: Arc<FiltersimParams>,
    filter: [ActiveLpf; CHANNELS],
    state: [ActiveLpfState; CHANNELS],
    oversample: [Oversample; CHANNELS],
}

#[derive(Params)]
struct FiltersimParams {
    #[id = "freq"]
    pub freq: FloatParam,
    #[id = "amp"]
    pub amp: FloatParam,
}

const BLOCK_SIZE: usize = 64;
const OVERSAMPLE: usize = 4;

impl<const C: usize> Default for Filtersim<C> {
    fn default() -> Self {
        Self {
            params: Arc::new(FiltersimParams::default()),
            filter: [ActiveLpf::new(300.0); C],
            state: [ActiveLpfState::default(); C],
            oversample: std::array::from_fn(|_| Oversample::new(OVERSAMPLE, BLOCK_SIZE)),
        }
    }
}

impl Default for FiltersimParams {
    fn default() -> Self {
        Self {
            freq: FloatParam::new(
                "Frequency",
                300.0,
                FloatRange::Skewed {
                    min: 20.0,
                    max: 30e3,
                    factor: FloatRange::skew_factor(-2.0),
                },
            )
            .with_smoother(SmoothingStyle::Linear(0.01))
            .with_value_to_string(formatters::v2s_f32_hz_then_khz(2))
            .with_string_to_value(formatters::s2v_f32_hz_then_khz()),
            amp: FloatParam::new(
                "Amp",
                1.0,
                FloatRange::Linear {
                    min: 0.0,
                    max: 10.0,
                },
            )
            .with_value_to_string(formatters::v2s_f32_gain_to_db(2))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
        }
    }
}

impl<const CHANNELS: usize> Plugin for Filtersim<CHANNELS> {
    const NAME: &'static str = "Filtersim";
    const VENDOR: &'static str = "SolarLiner";
    const URL: &'static str = "https://youtu.be/dQw4w9WgXcQ";
    const EMAIL: &'static str = "solarliner@gmail.com";

    const VERSION: &'static str = "0.0.1";

    const DEFAULT_INPUT_CHANNELS: u32 = CHANNELS as u32;
    const DEFAULT_OUTPUT_CHANNELS: u32 = CHANNELS as u32;

    const DEFAULT_AUX_INPUTS: Option<AuxiliaryIOConfig> = None;
    const DEFAULT_AUX_OUTPUTS: Option<AuxiliaryIOConfig> = None;

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn accepts_bus_config(&self, config: &BusConfig) -> bool {
        // This works with any symmetrical IO layout
        config.num_input_channels == config.num_output_channels && config.num_input_channels > 0
    }

    fn initialize(
        &mut self,
        _bus_config: &BusConfig,
        _buffer_config: &BufferConfig,
        context: &mut impl InitContext,
    ) -> bool {
        context.set_latency_samples(self.oversample[0].latency_samples());
        true
    }

    fn reset(&mut self) {
        // Reset buffers and envelopes here. This can be called from the audio thread and may not
        // allocate. You can remove this function if you do not need it.
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext,
    ) -> ProcessStatus {
        let sr = context.transport().sample_rate as f64;
        let os_sr = OVERSAMPLE as f64 * sr;
        let _os_sr_step = os_sr.recip();
        // Smoothing is optionally built into the parameters themselves
        let amp = self.params.amp.value();
        for filter in self.filter.iter_mut() {
            filter.set_amp(amp as _);
        }

        let mut f64_block = [0.; BLOCK_SIZE];
        for (_i, block) in buffer.iter_blocks(BLOCK_SIZE) {
            for (ch, block) in block.into_iter().enumerate() {
                for (s64, s) in f64_block.iter_mut().zip(block.iter().copied()) {
                    *s64 = s as _;
                }
                self.oversample[ch].with_oversample(&mut f64_block, |_data| {
                    /*                     for ele in data {
                        *ele = (*ele).clamp(-0.1, 0.1);
                    } */
                });
                for (s, s64) in block.iter_mut().zip(f64_block.iter().copied()) {
                    *s = s64 as _;
                }
            }
        }
        /*         self.stft.process_overlap_add(buffer, 1, |_ch, data| {
            let buflen = self.zero_stuff(data);
            self.conv.process(&mut self.os_buffer[..buflen]);
            for s in self.os_buffer.iter_mut().take(buflen) {
                *s = (*s).clamp(-0.1, 0.1);
            }
            self.conv.process(&mut self.os_buffer[..buflen]);
            self.decimate(data);
        }); */
        ProcessStatus::Normal
    }
}

impl<const CHANNELS: usize> ClapPlugin for Filtersim<CHANNELS> {
    const CLAP_ID: &'static str = "com.github.solarliner.filtersim";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Simulation of simple filter circuit from ODEs");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;

    // Don't forget to change these features
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Stereo];
}

impl<const CHANNELS: usize> Vst3Plugin for Filtersim<CHANNELS> {
    const VST3_CLASS_ID: [u8; 16] = *b"filtersimsolarln";

    // And don't forget to change these categories, see the docstring on `VST3_CATEGORIES` for more
    // information
    const VST3_CATEGORIES: &'static str = "Fx|Dynamics";
}

nih_export_clap!(Filtersim::<2>);
nih_export_vst3!(Filtersim::<2>);
