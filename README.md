ranged-num is a crate for representing a single number that is
in some compile-time known range of values. You can use any underlying
numeric type to hold the actual value. The main type to look at is
[Ranged](https://docs.rs/ranged-num/latest/ranged_num/struct.Ranged.html). 

Example:

```
use typenum::{N2, P3, P4};
use ranged_num::Ranged;

let ranged = Ranged::<N2, P4, isize>::new::<P3>();
assert_eq!(ranged.value(), 3);
```
