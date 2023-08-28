use crate::formats;
pub use rgb::alt::Gray;
pub use rgb::RGB;
pub use rgb::RGBA;
#[cfg(feature = "fp16")]
pub use half::f16;

/// This is a the floating-point type used for calculations.
#[cfg(not(feature = "fp16"))]
#[allow(non_camel_case_types)]
pub(crate) type fpN = f32;
#[cfg(feature = "fp16")]
#[allow(non_camel_case_types)]
pub(crate) type fpN = f16;

/// Use [`Pixel`](crate::Pixel) presets to specify pixel format.
///
/// The trait represents a temporary object that adds pixels together.
pub trait PixelFormat: Send + Sync {
    /// Pixel type in the source image
    type InputPixel: Send + Sync + Copy;
    /// Pixel type in the destination image (usually the same as Input)
    type OutputPixel: Default + Send + Sync + Copy;
    /// Temporary struct for the pixel in floating-point
    type Accumulator: Send + Sync + Copy;

    /// Create new floating-point pixel
    fn new() -> Self::Accumulator;
    /// Add new pixel with a given weight (first axis)
    fn add(&self, acc: &mut Self::Accumulator, inp: Self::InputPixel, coeff: fpN);
    /// Add bunch of accumulated pixels with a weight (second axis)
    fn add_acc(acc: &mut Self::Accumulator, inp: Self::Accumulator, coeff: fpN);
    /// Finalize, convert to output pixel format
    fn into_pixel(&self, acc: Self::Accumulator) -> Self::OutputPixel;
}

impl<F: ToFloat, T: ToFloat> PixelFormat for formats::Rgb<T, F> {
    type InputPixel = RGB<F>;
    type OutputPixel = RGB<T>;
    type Accumulator = RGB<fpN>;

    #[inline(always)]
    fn new() -> Self::Accumulator {
        RGB::new(fpN::ZERO, fpN::ZERO, fpN::ZERO)
    }

    #[inline(always)]
    fn add(&self, acc: &mut Self::Accumulator, inp: RGB<F>, coeff: fpN) {
        acc.r += inp.r.to_float() * coeff;
        acc.g += inp.g.to_float() * coeff;
        acc.b += inp.b.to_float() * coeff;
    }

    #[inline(always)]
    fn add_acc(acc: &mut Self::Accumulator, inp: Self::Accumulator, coeff: fpN) {
        acc.r += inp.r * coeff;
        acc.g += inp.g * coeff;
        acc.b += inp.b * coeff;
    }

    #[inline(always)]
    fn into_pixel(&self, acc: Self::Accumulator) -> RGB<T> {
        RGB {
            r: T::from_float(acc.r),
            g: T::from_float(acc.g),
            b: T::from_float(acc.b),
        }
    }
}

impl<F: ToFloat, T: ToFloat> PixelFormat for formats::Rgba<T, F> {
    type InputPixel = RGBA<F>;
    type OutputPixel = RGBA<T>;
    type Accumulator = RGBA<fpN>;

    #[inline(always)]
    fn new() -> Self::Accumulator {
        RGBA::new(fpN::ZERO, fpN::ZERO, fpN::ZERO, fpN::ZERO)
    }

    #[inline(always)]
    fn add(&self, acc: &mut Self::Accumulator, inp: RGBA<F>, coeff: fpN) {
        acc.r += inp.r.to_float() * coeff;
        acc.g += inp.g.to_float() * coeff;
        acc.b += inp.b.to_float() * coeff;
        acc.a += inp.a.to_float() * coeff;
    }

    #[inline(always)]
    fn add_acc(acc: &mut Self::Accumulator, inp: Self::Accumulator, coeff: fpN) {
        acc.r += inp.r * coeff;
        acc.g += inp.g * coeff;
        acc.b += inp.b * coeff;
        acc.a += inp.a * coeff;
    }

    #[inline(always)]
    fn into_pixel(&self, acc: Self::Accumulator) -> RGBA<T> {
        RGBA {
            r: T::from_float(acc.r),
            g: T::from_float(acc.g),
            b: T::from_float(acc.b),
            a: T::from_float(acc.a),
        }
    }
}

impl<F: ToFloat, T: ToFloat> PixelFormat for formats::RgbaPremultiply<T, F> {
    type InputPixel = RGBA<F>;
    type OutputPixel = RGBA<T>;
    type Accumulator = RGBA<fpN>;

    #[inline(always)]
    fn new() -> Self::Accumulator {
        RGBA::new(fpN::ZERO, fpN::ZERO, fpN::ZERO, fpN::ZERO)
    }

    #[inline(always)]
    fn add(&self, acc: &mut Self::Accumulator, inp: RGBA<F>, coeff: fpN) {
        let a_coeff = inp.a.to_float() * coeff;
        acc.r += inp.r.to_float() * a_coeff;
        acc.g += inp.g.to_float() * a_coeff;
        acc.b += inp.b.to_float() * a_coeff;
        acc.a += a_coeff;
    }

    #[inline(always)]
    fn add_acc(acc: &mut Self::Accumulator, inp: Self::Accumulator, coeff: fpN) {
        acc.r += inp.r * coeff;
        acc.g += inp.g * coeff;
        acc.b += inp.b * coeff;
        acc.a += inp.a * coeff;
    }

    #[inline(always)]
    fn into_pixel(&self, acc: Self::Accumulator) -> RGBA<T> {
        if acc.a > fpN::ZERO {
            let inv = fpN::ONE / acc.a;
            RGBA {
                r: T::from_float(acc.r * inv),
                g: T::from_float(acc.g * inv),
                b: T::from_float(acc.b * inv),
                a: T::from_float(acc.a),
            }
        } else {
            let zero = T::from_float(fpN::ZERO);
            RGBA::new(zero, zero, zero, zero)
        }
    }
}

impl<F: ToFloat, T: ToFloat> PixelFormat for formats::Gray<F, T> {
    type InputPixel = Gray<F>;
    type OutputPixel = Gray<T>;
    type Accumulator = Gray<fpN>;

    #[inline(always)]
    fn new() -> Self::Accumulator {
        Gray::new(fpN::ZERO)
    }

    #[inline(always)]
    fn add(&self, acc: &mut Self::Accumulator, inp: Gray<F>, coeff: fpN) {
        acc.0 += inp.0.to_float() * coeff;
    }

    #[inline(always)]
    fn add_acc(acc: &mut Self::Accumulator, inp: Self::Accumulator, coeff: fpN) {
        acc.0 += inp.0 * coeff;
    }

    #[inline(always)]
    fn into_pixel(&self, acc: Self::Accumulator) -> Gray<T> {
        Gray::new(T::from_float(acc.0))
    }
}

use self::f::ToFloat;
mod f {
    use super::fpN;
    use num_traits::{FromPrimitive, ToPrimitive, AsPrimitive};


    /// Internal, please don't use
    pub trait ToFloat: Default + Send + Sync + Sized + Copy + 'static {
        fn to_float(self) -> fpN;
        fn from_float(f: fpN) -> Self;
    }

    impl ToFloat for u8 {
        #[inline(always)]
        fn to_float(self) -> fpN {
            fpN::from(self)
        }

        #[inline(always)]
        fn from_float(f: fpN) -> Self {
            let r: u8 = (f + fpN::from_f32(0.5)).as_();
            r
        }
    }

    impl ToFloat for u16 {
        #[inline(always)]
        fn to_float(self) -> fpN {
            fpN::from_f32(self as f32)
        }

        #[inline(always)]
        fn from_float(f: fpN) -> Self {
            let r: u16 = (f + fpN::from_f32(0.5)).as_();
            r
        }
    }

    impl ToFloat for fpN {
        #[inline(always)]
        fn to_float(self) -> fpN {
            self
        }

        #[inline(always)]
        fn from_float(f: fpN) -> Self {
            f
        }
    }

    impl ToFloat for f64 {
        #[inline(always)]
        fn to_float(self) -> fpN {
            fpN::from_f64(self)
        }

        #[inline(always)]
        fn from_float(f: fpN) -> Self {
            f.to_f64()
        }
    }
}
