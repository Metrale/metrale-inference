# Layouts as Functions

Kernels index tiles: which element of a matrix a thread loads, where in shared memory it is
staged, and which register fragment it feeds. The crate `metrale-layout` (`crates/layout`)
states those index maps as functions, so a build script or a test can compose and check them on
the CPU before any kernel runs. It is pure arithmetic, with no dependencies and no I/O.

## Layouts

A **layout** is a list of modes. Each mode is a list of digits `(size, stride)`. An index `i`
in `[0, size)` is expanded into mixed-radix digits, first digit fastest:

```text
digit_k(i) = floor(i / (s_1 ... s_{k-1})) mod s_k
f(i)       = Σ_k digit_k(i) · d_k
```

This is the positional number system with mixed radices (Knuth, *The Art of Computer
Programming*, vol. 2, §4.1), followed by an integer linear form. Two layouts are equal when their
functions are equal. A row-major `4 × 8` matrix is `(4, 8), (8, 1)`, and element `(1, 2)` is at
offset `1·8 + 2·1 = 10`.

The **cosize** is one past the largest offset, which is the extent the layout addresses.

## Operations

| Operation | Meaning |
|---|---|
| `coalesce(L)` | merge digits that continue each other (`d_{k+1} = s_k d_k`) and drop size-1 digits. The function is unchanged |
| `compose(A, B)` | the layout of `A ∘ B`: index `i` maps to `A(B(i))`, for a `B` that stays inside `A`'s domain |
| `complement(L, M)` | the offsets `L` leaves out of `[0, M)`, in order, so that `(L, complement)` is a bijection onto `[0, M)` |
| `divide(L, T)` | `L ∘ (T, complement(T, size(L)))`: mode 0 indexes inside a tile and mode 1 indexes the tiles |
| `product(L, T)` | `(L, complement(L, size(L)·cosize(T)) ∘ T)`: `L` repeated as `T` arranges the copies |

Tiling a matrix among warps, then among the threads of a warp, is a chain of `divide` and
`compose`.

**Every result is checked by evaluation.** Each operation computes a closed form and then
compares it with the function it stands for over the whole domain. When no layout equals that
function (a stride that does not divide the digit sizes it steps over), the operation returns an
error. It never returns a wrong layout.

## Swizzles

A **swizzle** `σ(b, m, s)` XORs the `b`-bit field at bit `m + s` of an offset into the field at
bit `m`:

```text
σ(o) = o ⊕ (((o >> (m + s)) & (2^b − 1)) << m)
```

On the bits of the offset this is a linear map over GF(2) that is its own inverse. It therefore
permutes offsets within each aligned block of `2^(m+s+b)` and moves none across blocks.
Shared-memory tiles use it so that the rows one warp reads land in different banks. For example,
gb10's FP8 GEMM pipeline swizzles 16-byte chunks of 64-byte rows by `chunk ⊕ ((row >> 1) & 3)`,
which is `σ(2, 4, 3)`.

## Checks

- **Bijectivity and injectivity**, over the layout's domain.
- **Bank conflicts.** For each phase of a warp's shared-memory access, the check counts the
  distinct 4-byte words any one of the 32 banks serves. A phase is the lanes that move 128 bytes.
  The result is 1 when the access is conflict-free. Unswizzled 64-byte rows read 16 bytes per
  lane are 4-way conflicted; with `σ(2, 4, 3)` they are conflict-free.
- **Vector alignment.** Each lane's vector access starts on a multiple of its width.

## Scope

The crate covers only what the kernels need: one level of modes over digits, integer strides,
and XOR swizzles. It grows by the same promotion rule as the kernel blueprint. An operation joins
when a second kernel needs it, and not before.
