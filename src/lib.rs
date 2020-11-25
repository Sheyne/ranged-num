#![no_std]

//! ranged-num is a crate for representing a single number that is
//! in some compile-time known range of values. You can use any underlying
//! numeric type to hold the actual value. The main type to look at is
//! [Ranged](struct.Ranged.html). 
//! 
//! Example:
//! 
//! ```
//! use typenum::{N2, P3, P4};
//! use ranged_num::Ranged;
//!
//! let ranged = Ranged::<N2, P4, isize>::new::<P3>();
//! assert_eq!(ranged.value(), 3);
//! ```

use core::marker::PhantomData;
#[macro_use]
extern crate typenum;
use core::ops::{Add, Mul, Rem, Sub};
use typenum::{op, Integer, IsLessOrEqual, Max, Min, NInt, NonZero, PInt, UInt, Unsigned, U0, U1};

/// These are used by a macro. In order for them to be accessible to the macro
/// I re-exported them. If you know of a better way to do this, let me know.  
pub mod reexports {
    pub use typenum::{type_operators::IsLessOrEqual, Unsigned, U0, U1};
}

/// Holds a single value between `Min` and `Max`. That value is stored in a single
/// variable of type `Underlying`.
/// 
/// Example:
/// ```
/// use typenum::{N5, P5};
/// use ranged_num::Ranged;
///
/// let ranged = Ranged::<N5, P5, i8>::try_new(-1).unwrap();
/// assert_eq!(ranged.value(), -1);
#[derive(Debug)]
pub struct Ranged<Min, Max, Underlying>(Underlying, PhantomData<Min>, PhantomData<Max>);

/// Represent a usize in `[0, Max]` (inclusive).
/// 
/// Example:
/// ```
/// use typenum::U5;
/// use ranged_num::UZeroTo;
/// 
/// let x: UZeroTo<U5> = UZeroTo::try_new(4).unwrap();
/// ```
pub type UZeroTo<Max> = Ranged<U0, Max, usize>;

/// `typenum` doesn't currently allow you to easily specify any numbers
/// (signed or unsigned) that can be converted to a given type. I've added
/// `CanMake<V>` as a trait to indicate which types can easily make a runtime
/// value of type `V` to use in type bounds. 
pub trait CanMake<V> {
    /// Create the `V` that goes with this type. 
    /// 
    /// Example:
    /// 
    /// ```
    /// use typenum::P5;
    /// use ranged_num::CanMake;
    /// 
    /// let x: i8 = <P5 as CanMake<i8>>::make();
    /// assert_eq!(x, 5);
    /// ```
    fn make() -> V;
}

/// `typenum` doesn't currently allow you to easily add an `Integer` with an `Unsigned`.
/// The `AddU` trait lets you add some `Unsigned` to both `Integer`s and `Unsigned`s.
/// 
/// Example:
/// 
/// ```
/// use typenum::*;
/// use ranged_num::AddU;
/// 
/// assert_type_eq!(<U6 as AddU<U3>>::Output, U9);
/// assert_type_eq!(<P6 as AddU<U3>>::Output, P9);
/// assert_type_eq!(<N6 as AddU<U3>>::Output, N3);
/// ```
pub trait AddU<U>
where
    U: Unsigned,
{
    type Output;
}

impl<U, B, T> AddU<T> for UInt<U, B>
where
    T: Unsigned,
    UInt<U, B>: Add<T>,
{
    type Output = <UInt<U, B> as Add<T>>::Output;
}

impl<U, T> AddU<T> for PInt<U>
where
    T: Unsigned + NonZero,
    U: Unsigned + NonZero,
    PInt<U>: Add<PInt<T>>,
{
    type Output = <PInt<U> as Add<PInt<T>>>::Output;
}

impl<U, T> AddU<T> for NInt<U>
where
    T: Unsigned + NonZero,
    U: Unsigned + NonZero,
    NInt<U>: Add<PInt<T>>,
{
    type Output = <NInt<U> as Add<PInt<T>>>::Output;
}

macro_rules! define_can_make {
    ($thing:ident => $type:ty : $assoc:ident) => {
        impl<T> CanMake<$type> for T
        where
            T: $thing,
        {
            fn make() -> $type {
                T::$assoc.into()
            }
        }
    };
}

define_can_make!(Integer => isize : ISIZE);
define_can_make!(Integer => i8 : I8);
define_can_make!(Integer => i16 : I16);
define_can_make!(Integer => i32 : I32);
define_can_make!(Integer => i64 : I64);
define_can_make!(Integer => f32 : I16);
define_can_make!(Integer => f64 : I32);

define_can_make!(Unsigned => usize : USIZE);
define_can_make!(Unsigned => u8 : U8);
define_can_make!(Unsigned => u16 : U16);
define_can_make!(Unsigned => u32 : U32);
define_can_make!(Unsigned => u64 : U64);

impl<Min, Max, Underlying> Ranged<Min, Max, Underlying> {
    /// Create a new `Ranged` holding the compile time known `T`.
    ///
    /// ```
    /// use typenum::{N2, P3, P4};
    /// use ranged_num::Ranged;
    ///
    /// let ranged = Ranged::<N2, P4, isize>::new::<P3>();
    /// assert_eq!(ranged.value(), 3);
    /// ```
    pub fn new<T>() -> Self
    where
        T: IsLessOrEqual<Max> + CanMake<Underlying>,
        Min: IsLessOrEqual<T>,
    {
        Ranged(T::make(), PhantomData, PhantomData)
    }

    /// Create a new `Randed` from `Min` to `Max` (inclusive) from the given value.
    /// Returns Some(Ranged) if the given value is between `Min` and `Max`.
    ///
    /// Example:
    /// ```
    /// use typenum::{U2, U4};
    /// use ranged_num::Ranged;
    ///
    /// let ranged = Ranged::<U2, U4, usize>::try_new(3).unwrap();
    /// assert_eq!(ranged.value(), 3);
    ///
    /// let x = Ranged::<U2, U4, usize>::try_new(1);
    /// assert!(x.is_none());
    /// ```
    pub fn try_new(u: Underlying) -> Option<Self>
    where
        Min: CanMake<Underlying>,
        Max: CanMake<Underlying>,
        Underlying: PartialOrd<Underlying>,
    {
        if u < Min::make() {
            None
        } else if u > Max::make() {
            None
        } else {
            Some(Ranged(u, PhantomData, PhantomData))
        }
    }

    /// Add `u` to the value held by this `Ranged`. If the sum wraps past
    /// `Max` start again from `Min`.
    ///
    /// Example:
    /// ```
    /// use typenum::{N2, P2, P4};
    /// use ranged_num::Ranged;
    ///
    /// let x = Ranged::<N2, P4, isize>::new::<P2>();
    /// let y: Ranged<N2, P4, isize> = x.wrapped_add(4);
    /// assert_eq!(y.value(), -1);
    /// ```
    pub fn wrapped_add(&self, u: Underlying) -> Self
    where
        Min: CanMake<Underlying>,
        Max: CanMake<Underlying> + AddU<U1>,
        <Max as AddU<U1>>::Output: Sub<Min>,
        <<Max as AddU<U1>>::Output as Sub<Min>>::Output: CanMake<Underlying>,
        Underlying: Copy
            + Add<Underlying, Output = Underlying>
            + Sub<Underlying, Output = Underlying>
            + Rem<Underlying, Output = Underlying>,
    {
        let sum = self.0 + u;
        let range = <<Max as AddU<U1>>::Output as Sub<Min>>::Output::make();
        Ranged(
            ((sum - Min::make()) % range) + Min::make(),
            PhantomData,
            PhantomData,
        )
    }

    /// Offset the `Ranged` by `T`.
    ///
    /// Example:
    /// ```
    /// use typenum::*;
    /// use ranged_num::Ranged;
    ///
    /// let x = Ranged::<N2, P4, isize>::new::<P2>();
    /// assert_eq!(x.value(), 2);
    ///
    /// let x: Ranged<N3, P3, isize> = x.offset::<N1>();
    /// assert_eq!(x.value(), 1);
    /// ```
    pub fn offset<T>(&self) -> Ranged<op!(Min + T), op!(Max + T), Underlying>
    where
        Min: Add<T>,
        Max: Add<T>,
        T: CanMake<Underlying>,
        Underlying: Copy + Add<Underlying, Output = Underlying>,
    {
        Ranged(self.value() + T::make(), PhantomData, PhantomData)
    }

    /// Retrieve the value of this Ranged. It will be between Min and Max (inclusive).
    ///
    /// Example:
    /// ```
    /// use typenum::{N2, P3, P4};
    /// use ranged_num::Ranged;
    ///
    /// let ranged = Ranged::<N2, P4, isize>::new::<P3>();
    /// assert_eq!(ranged.value(), 3);
    /// ```
    pub fn value(&self) -> Underlying
    where
        Underlying: Copy,
    {
        self.0
    }

    /// Calculate the maximum value that this Ranged could hold. That is
    /// get the runtime value of Max.
    ///
    /// Example:
    /// ```
    /// use typenum::{N2, P3, P4};
    /// use ranged_num::Ranged;
    ///
    /// let ranged = Ranged::<N2, P4, isize>::new::<P3>();
    /// assert_eq!(ranged.max_value(), 4);
    /// ```
    pub fn max_value(&self) -> Underlying
    where
        Max: CanMake<Underlying>,
    {
        Max::make()
    }

    /// Calculate the minimum value that this Ranged could hold. That is
    /// get the runtime value of Min.
    ///
    /// Example:
    /// ```
    /// use typenum::{N2, P3, P4};
    /// use ranged_num::Ranged;
    ///
    /// let ranged = Ranged::<N2, P4, isize>::new::<P3>();
    /// assert_eq!(ranged.min_value(), -2);
    /// ```
    pub fn min_value(&self) -> Underlying
    where
        Min: CanMake<Underlying>,
    {
        Min::make()
    }
}

impl<Min1, Max1, Min2, Max2, Underlying> Add<Ranged<Min2, Max2, Underlying>>
    for Ranged<Min1, Max1, Underlying>
where
    Min1: Add<Min2>,
    Max1: Add<Max2>,
    Underlying: Add<Underlying, Output = Underlying>,
{
    type Output = Ranged<op!(Min1 + Min2), op!(Max1 + Max2), Underlying>;

    fn add(self, other: Ranged<Min2, Max2, Underlying>) -> Self::Output {
        Ranged(self.0 + other.0, PhantomData, PhantomData)
    }
}

impl<Min1, Max1, Min2, Max2, Underlying> Sub<Ranged<Min2, Max2, Underlying>>
    for Ranged<Min1, Max1, Underlying>
where
    Min1: Sub<Max2>,
    Max1: Sub<Min2>,
    Underlying: Sub<Underlying, Output = Underlying>,
{
    type Output = Ranged<op!(Min1 - Max2), op!(Max1 - Min2), Underlying>;

    fn sub(self, other: Ranged<Min2, Max2, Underlying>) -> Self::Output {
        Ranged(self.0 - other.0, PhantomData, PhantomData)
    }
}

impl<Min1, Max1, Min2, Max2, Underlying> Mul<Ranged<Min2, Max2, Underlying>>
    for Ranged<Min1, Max1, Underlying>
where
    Min1: Mul<Min2> + Mul<Max2>,
    Max1: Mul<Min2> + Mul<Max2>,
    op!(Min1 * Min2): Min<op!(Min1 * Max2)>,
    op!(Max1 * Min2): Min<op!(Max1 * Max2)>,
    op!(min(Min1 * Min2, Min1 * Max2)): Min<op!(min(Max1 * Min2, Max1 * Max2))>,
    op!(Min1 * Min2): Max<op!(Min1 * Max2)>,
    op!(Max1 * Min2): Max<op!(Max1 * Max2)>,
    op!(max(Min1 * Min2, Min1 * Max2)): Max<op!(max(Max1 * Min2, Max1 * Max2))>,
    Underlying: Mul<Underlying, Output = Underlying>,
{
    type Output = Ranged<
        op!(min(
            min(Min1 * Min2, Min1 * Max2),
            min(Max1 * Min2, Max1 * Max2)
        )),
        op!(max(
            max(Min1 * Min2, Min1 * Max2),
            max(Max1 * Min2, Max1 * Max2)
        )),
        Underlying,
    >;

    fn mul(self, other: Ranged<Min2, Max2, Underlying>) -> Self::Output {
        Ranged(self.0 * other.0, PhantomData, PhantomData)
    }
}

impl<Min1, Max1, Min2, Max2, Underlying> Rem<Ranged<Min2, Max2, Underlying>>
    for Ranged<Min1, Max1, Underlying>
where
    Underlying: Rem<Underlying, Output = Underlying>,
    Max2: Sub<U1>,
{
    type Output = Ranged<U0, op!(Max2 - U1), Underlying>;

    fn rem(self, other: Ranged<Min2, Max2, Underlying>) -> Self::Output {
        Ranged(self.0 % other.0, PhantomData, PhantomData)
    }
}

/// Define a new enum which can be convered to and from a Ranged usize.
///
/// Example:
/// ```
/// use ranged_num::{define_ranged_enum, UZeroTo};
///
/// define_ranged_enum!(Example, Derive(Debug, PartialEq, Eq), A, B, C);
///
/// let x: UZeroTo<_> = Example::A.into();
/// assert_eq!(x.value(), 0);
///
/// let x = x.wrapped_add(5);
///
/// let y: Example = x.into();
/// assert_eq!(y, Example::C);
/// ```
#[macro_export]
macro_rules! define_ranged_enum {
    (@inner_count ; $sum:ty ; $fst:ident, ) => {
        $sum
    };
    (@inner_count ; $sum:ty ; $fst:ident, $($rest:ident,)*) => {
        $crate::define_ranged_enum!(@inner_count ; <$sum as ::core::ops::Add<$crate::reexports::U1>>::Output ; $($rest,)*)
    };
    (@inner_match_from ; $idx:expr; $sum:ty ; $($i:ty : $arm:ident)*; ) => {
        match $idx {
            $(<$i as $crate::reexports::Unsigned>::USIZE => Self::$arm,)*
            _ => unreachable!()
        }
    };
    (@inner_match_from ; $idx:expr; $sum:ty ; $($i:ty : $arm:ident)*; $fst:ident, $($rest:ident,)*) => {
        $crate::define_ranged_enum!(@inner_match_from ; $idx;
            <$sum as ::core::ops::Add<$crate::reexports::U1>>::Output ;
            $($i : $arm)* $sum : $fst;
            $($rest,)*)
    };
    (@inner_match_to ; $name:ident; $idx:expr; $sum:ty ; $($i:ty : $arm:ident)*; ) => {
        match $idx {
            $($name::$arm => <$i as $crate::reexports::Unsigned>::USIZE,)*
        }
    };
    (@inner_match_to ; $name:ident; $idx:expr; $sum:ty ; $($i:ty : $arm:ident)*; $fst:ident, $($rest:ident,)*) => {
        $crate::define_ranged_enum!(@inner_match_to; $name ; $idx;
            <$sum as ::core::ops::Add<$crate::reexports::U1>>::Output ;
            $($i : $arm)* $sum : $fst;
            $($rest,)*)
    };
    ($name:ident, Derive($($derive:ident),* $(,)?), $($x:ident),+ $(,)?) => {
        #[derive($($derive),*)]
        pub enum $name {
            $($x),+
        }

        impl From<$crate::Ranged<$crate::reexports::U0, define_ranged_enum!(@inner_count; $crate::reexports::U0; $($x,)+), usize>> for $name {
            fn from(x: $crate::Ranged<$crate::reexports::U0, define_ranged_enum!(@inner_count; $crate::reexports::U0; $($x,)+), usize>)
            -> Self {
                $crate::define_ranged_enum!(@inner_match_from; x.value(); $crate::reexports::U0; ; $($x,)+)
            }
        }

        impl From<$name> for $crate::Ranged<$crate::reexports::U0, define_ranged_enum!(@inner_count; $crate::reexports::U0; $($x,)+), usize> {
            fn from(x: $name)
            -> $crate::Ranged<$crate::reexports::U0, define_ranged_enum!(@inner_count; $crate::reexports::U0; $($x,)+), usize> {
                $crate::Ranged::<$crate::reexports::U0, define_ranged_enum!(@inner_count; $crate::reexports::U0; $($x,)+), usize>::try_new($crate::define_ranged_enum!(@inner_match_to; $name; x; $crate::reexports::U0; ; $($x,)+)).unwrap()
            }
        }
    };
    ($name:ident, $($x:ident),+ $(,)?) => {
        $crate::define_ranged_enum!($name, Derive(), $($x),+);
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use typenum::*;

    #[test]
    fn cant_over() {
        let x = Ranged::<U2, U4, u8>::try_new(5);
        assert!(x.is_none());
    }

    #[test]
    fn add_bounds() {
        let x = Ranged::<N2, P4, isize>::new::<P4>();
        let y = Ranged::<N3, P4, isize>::new::<P4>();
        let sum: Ranged<N5, P8, isize> = x + y;
        assert_eq!(sum.value(), 8);
    }

    #[test]
    fn sub_bounds() {
        let x = Ranged::<N2, P4, isize>::new::<P3>();
        let y = Ranged::<N3, P4, isize>::new::<P4>();
        let sum: Ranged<N6, P7, isize> = x - y;
        assert_eq!(sum.value(), -1);
    }

    #[test]
    fn mul_bounds() {
        let x = Ranged::<N2, P4, isize>::new::<P4>();
        let y = Ranged::<N3, P4, isize>::new::<P4>();
        let prod: Ranged<N12, P16, isize> = x * y;
        assert_eq!(prod.value(), 16);
    }

    #[test]
    fn mod_bounds() {
        let x = Ranged::<U3, U8, usize>::new::<U4>();
        let y = Ranged::<U2, U4, usize>::new::<U3>();
        let rem: Ranged<U0, U3, usize> = x % y;
        assert_eq!(rem.value(), 1);
    }

    #[test]
    fn add_bounds_u() {
        let x = Ranged::<U2, U4, usize>::new::<U4>();
        let y = Ranged::<U3, U4, usize>::new::<U4>();
        let sum: Ranged<U5, U8, usize> = x + y;
        assert_eq!(sum.value(), 8);
    }

    #[test]
    fn add_const() {
        let x = Ranged::<U1, U4, usize>::new::<U4>();
        let y: Ranged<U2, U5, usize> = x.offset::<U1>();
        assert_eq!(y.value(), 5);
    }

    #[test]
    fn wrapped_add_no_wrap() {
        let x = Ranged::<N2, P4, isize>::new::<P1>();
        let y: Ranged<N2, P4, isize> = x.wrapped_add(2);
        assert_eq!(y.value(), 3);
    }

    #[test]
    fn wrapped_add_overflow() {
        let x = Ranged::<N2, P4, isize>::new::<P1>();
        let y: Ranged<N2, P4, isize> = x.wrapped_add(4);
        assert_eq!(y.value(), -2);
    }

    #[test]
    fn define_ranged_enum_test() {
        define_ranged_enum!(RangedEnumTest, Derive(Debug, PartialEq, Eq), A, B, C);

        let x = RangedEnumTest::from(UZeroTo::new::<U2>());
        assert_eq!(x, RangedEnumTest::C);
    }

    #[test]
    fn define_ranged_enum_test_2() {
        define_ranged_enum!(RangedEnumTest, Derive(Debug, PartialEq, Eq), A, B, C);

        let x: UZeroTo<U2> = RangedEnumTest::C.into();
        assert_eq!(x.value(), 2);
    }

    #[test]
    fn define_ranged_enum_test_3() {
        define_ranged_enum!(Example, Derive(Debug, PartialEq, Eq), A, B, C);

        let x: UZeroTo<_> = Example::A.into();
        assert_eq!(x.value(), 0);

        let x = x.wrapped_add(5);

        let y: Example = x.into();
        assert_eq!(y, Example::C);
    }

    #[test]
    fn test_add_u() {
        assert_type_eq!(<P2 as AddU<U3>>::Output, P5);
        assert_type_eq!(<N2 as AddU<U3>>::Output, P1);
        assert_type_eq!(<U2 as AddU<U3>>::Output, U5);
    }

    #[test]
    fn test_bounded_float() {
        let x = Ranged::<N3, P5, f64>::try_new(4.5).unwrap();
        let y = Ranged::<P1, P2, f64>::try_new(1.6).unwrap();

        let sum = x + y;

        assert_eq!(sum.max_value(), 7f64);
        assert_eq!(sum.min_value(), -2f64);
        assert!((sum.value() - (4.5 + 1.6)).abs() < 0.001);

        assert!(Ranged::<N3, P5, f64>::try_new(-4.5).is_none());
    }
}
