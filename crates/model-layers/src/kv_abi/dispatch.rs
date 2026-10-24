// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The one match from a codec's launch-argument family (`AbiId`) to its zero-sized
//! `KvAbi` type. Every executor reaches its family through `with_abi`.
//!
//! Owner: model-layers.
//! Invariants: the match lists every `AbiId` and has no wildcard arm, so a new family fails to
//! compile here until it has a type.

use metrale_error::Error;
use metrale_kv_codec::{AbiId, KvCodecSpec};

use super::{
    BlockScaled, Dense, DenseScaled, KvAbi, KvExecutor, KvSideArgs, Latent, SideSource, Trellis,
};

/// 2026-10-18: A computation generic over the family, run by `with_abi` for one `AbiId`.
pub trait AbiFn<R> {
    /// 2026-10-18: Run for family `A`.
    fn call<A: KvAbi>(self) -> R;
}

/// 2026-10-18: Runs `f` for the family `abi` names.
#[deny(clippy::wildcard_enum_match_arm)]
pub fn with_abi<R>(abi: AbiId, f: impl AbiFn<R>) -> R {
    match abi {
        AbiId::Dense => f.call::<Dense>(),
        AbiId::DenseScaled => f.call::<DenseScaled>(),
        AbiId::BlockScaled => f.call::<BlockScaled>(),
        AbiId::Trellis => f.call::<Trellis>(),
        AbiId::Latent => f.call::<Latent>(),
    }
}

/// 2026-10-18: The side arguments of a side stored as `spec`, from its family.
pub fn side_args(spec: &'static KvCodecSpec, src: SideSource) -> Result<KvSideArgs, Error> {
    struct SideArgs(&'static KvCodecSpec, SideSource);
    impl AbiFn<Result<KvSideArgs, Error>> for SideArgs {
        fn call<A: KvAbi>(self) -> Result<KvSideArgs, Error> {
            A::side_args(self.0, self.1)
        }
    }
    with_abi(spec.abi, SideArgs(spec, src))
}

/// 2026-10-18: Decode attention over a symmetric cache of family `abi`.
pub fn decode<E: KvExecutor + ?Sized>(abi: AbiId, e: &E, a: &E::Decode<'_>) -> anyhow::Result<()> {
    struct Decode<'a, 'b, E: KvExecutor + ?Sized>(&'a E, &'a E::Decode<'b>);
    impl<E: KvExecutor + ?Sized> AbiFn<anyhow::Result<()>> for Decode<'_, '_, E> {
        fn call<A: KvAbi>(self) -> anyhow::Result<()> {
            A::decode(self.0, self.1)
        }
    }
    with_abi(abi, Decode(e, a))
}

/// 2026-10-18: Write into a symmetric cache of family `abi`.
pub fn write<E: KvExecutor + ?Sized>(abi: AbiId, e: &E, a: &E::Write<'_>) -> anyhow::Result<()> {
    struct Write<'a, 'b, E: KvExecutor + ?Sized>(&'a E, &'a E::Write<'b>);
    impl<E: KvExecutor + ?Sized> AbiFn<anyhow::Result<()>> for Write<'_, '_, E> {
        fn call<A: KvAbi>(self) -> anyhow::Result<()> {
            A::write(self.0, self.1)
        }
    }
    with_abi(abi, Write(e, a))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Id;
    impl AbiFn<AbiId> for Id {
        fn call<A: KvAbi>(self) -> AbiId {
            A::ID
        }
    }

    /// 2026-10-18: Each family reaches the type that stands for it, so no arm is crossed.
    #[test]
    fn every_family_reaches_its_own_type() {
        for abi in [
            AbiId::Dense,
            AbiId::DenseScaled,
            AbiId::BlockScaled,
            AbiId::Trellis,
            AbiId::Latent,
        ] {
            assert_eq!(with_abi(abi, Id), abi);
        }
    }

    /// 2026-10-18: Records which executor method ran.
    #[derive(Default)]
    struct Rec(core::cell::RefCell<Vec<&'static str>>);

    impl Rec {
        fn ran(&self, m: &'static str) -> anyhow::Result<()> {
            self.0.borrow_mut().push(m);
            Ok(())
        }
    }

    impl KvExecutor for Rec {
        type Decode<'a> = ();
        type Write<'a> = ();
        fn decode_dense(&self, _: &()) -> anyhow::Result<()> {
            self.ran("decode_dense")
        }
        fn decode_dense_scaled(&self, _: &()) -> anyhow::Result<()> {
            self.ran("decode_dense_scaled")
        }
        fn decode_block_scaled(&self, _: &()) -> anyhow::Result<()> {
            self.ran("decode_block_scaled")
        }
        fn decode_trellis(&self, _: &()) -> anyhow::Result<()> {
            self.ran("decode_trellis")
        }
        fn write_dense(&self, _: &()) -> anyhow::Result<()> {
            self.ran("write_dense")
        }
        fn write_dense_scaled(&self, _: &()) -> anyhow::Result<()> {
            self.ran("write_dense_scaled")
        }
        fn write_block_scaled(&self, _: &()) -> anyhow::Result<()> {
            self.ran("write_block_scaled")
        }
        fn write_trellis(&self, _: &()) -> anyhow::Result<()> {
            self.ran("write_trellis")
        }
    }

    /// 2026-10-18: Each implemented family runs its own decode and write method, and only it.
    #[test]
    fn each_family_runs_its_own_executor_methods() {
        for (abi, want) in [
            (AbiId::Dense, ["decode_dense", "write_dense"]),
            (
                AbiId::DenseScaled,
                ["decode_dense_scaled", "write_dense_scaled"],
            ),
            (
                AbiId::BlockScaled,
                ["decode_block_scaled", "write_block_scaled"],
            ),
            (AbiId::Trellis, ["decode_trellis", "write_trellis"]),
        ] {
            let rec = Rec::default();
            decode(abi, &rec, &()).unwrap();
            write(abi, &rec, &()).unwrap();
            assert_eq!(rec.0.into_inner(), want, "{abi:?}");
        }
    }

    /// 2026-10-18: A declared family with no launcher refuses decode, write and side arguments
    /// with `KvAbiUnimplemented`, and never reaches the executor.
    #[test]
    fn declared_families_refuse_before_the_executor() {
        use metrale_error::ErrorCode;
        let code = |e: anyhow::Error| e.downcast_ref::<Error>().map(Error::code);
        let src = SideSource {
            pool: metrale_gpu_runtime::gpu::DevicePtr(1),
            block_stride: 2,
            data_bytes: 3,
            scale: 1.0,
        };
        let any_spec = metrale_kv_codec::spec(metrale_kv_codec::KvCodecId::ALL[0]);
        let abi = AbiId::Latent;
        let rec = Rec::default();
        let want = Some(ErrorCode::KvAbiUnimplemented);
        assert_eq!(code(decode(abi, &rec, &()).unwrap_err()), want);
        assert_eq!(code(write(abi, &rec, &()).unwrap_err()), want);
        assert!(rec.0.into_inner().is_empty());
        let side = with_abi(abi, SideArgsOf(any_spec, src)).unwrap_err();
        assert_eq!(side.code(), ErrorCode::KvAbiUnimplemented);
    }

    /// 2026-10-18: Every registry codec of the `Trellis` family builds `Trellis` side arguments
    /// that keep the pool and block stride (its kernels take nothing else per side), and no
    /// other family's codec does.
    #[test]
    fn trellis_codecs_and_only_they_build_trellis_side_arguments() {
        let src = SideSource {
            pool: metrale_gpu_runtime::gpu::DevicePtr(0x40),
            block_stride: 6656,
            data_bytes: 77,
            scale: 0.5,
        };
        let mut trellis = 0;
        for id in metrale_kv_codec::KvCodecId::ATTENTION {
            let s = metrale_kv_codec::spec(id);
            let built = side_args(s, src);
            if s.abi == AbiId::Trellis {
                trellis += 1;
                let args = built.unwrap_or_else(|e| panic!("{}: {e}", s.name));
                assert_eq!(args.trellis(), Some((src.pool, 6656)), "{}", s.name);
                assert_eq!(args.pool(), src.pool, "{}", s.name);
            } else if let Ok(args) = built {
                assert_eq!(args.trellis(), None, "{}", s.name);
            }
        }
        assert!(trellis >= 1, "the registry has no Trellis codec to check");
    }

    /// 2026-10-18: `side_args` for a family chosen by the caller, not by the spec.
    struct SideArgsOf(&'static KvCodecSpec, SideSource);
    impl AbiFn<Result<KvSideArgs, Error>> for SideArgsOf {
        fn call<A: KvAbi>(self) -> Result<KvSideArgs, Error> {
            A::side_args(self.0, self.1)
        }
    }
}
